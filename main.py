import asyncio
import pathlib
import argparse
import tensorflow as tf
import numpy as np
from PIL import Image
import pytesseract
import database

# Carrega o modelo treinado
MODEL_PATH = "plate_classifier.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Nomes das classes (devem corresponder às pastas do dataset)
# O TensorFlow ordena alfabeticamente, então 'invalidos' é 0 e 'validos' é 1
CLASS_NAMES = ['invalidos', 'validos']

# Parâmetros de imagem (devem ser os mesmos do treinamento)
IMG_HEIGHT = 150
IMG_WIDTH = 150

async def process_image(image_path: pathlib.Path):
    """Classifica uma imagem, executa OCR se for válida e salva no banco."""
    print(f"Processando: {image_path.name}...")
    
    # Carrega e pré-processa a imagem
    img = tf.keras.utils.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Cria um batch

    # Faz a predição
    predictions = model.predict(img_array)
    score = tf.nn.sigmoid(predictions[0])
    
    classification = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    ocr_text = None
    if classification == 'validos':
        print(f"  -> Classificado como VÁLIDO ({confidence:.2f}% de confiança). Executando OCR...")
        try:
            # Executa o OCR na imagem original para melhor qualidade
            ocr_text = pytesseract.image_to_string(Image.open(image_path))
            ocr_text = ocr_text.strip() if ocr_text else "Nenhum texto encontrado"
            print(f"  -> OCR Result: {ocr_text}")
        except Exception as e:
            print(f"Erro no OCR: {e}")
            ocr_text = f"Erro no OCR: {e}"
    else:
        print(f"  -> Classificado como INVÁLIDO ({confidence:.2f}% de confiança).")

    # Adiciona o registro ao banco de dados
    await database.add_record(
        image_path=str(image_path),
        classification=classification,
        ocr_text=ocr_text
    )
    print(f"  -> Resultado salvo no banco de dados.")

async def main(image_folder: str, limit: int = None):
    """Função principal para orquestrar o processo."""
    # Inicializa o banco de dados
    await database.init_db()
    print(f"Banco de dados '{database.DB_NAME}' inicializado.")

    # Define o diretório de entrada
    input_dir = pathlib.Path(image_folder)
    if not input_dir.is_dir():
        print(f"Erro: O diretório '{image_folder}' não foi encontrado.")
        return

    # Coleta todas as imagens e aplica o limite se fornecido
    all_images = list(input_dir.glob('**/*.jpg'))
    images_to_process = all_images[:limit] if limit else all_images
    
    print(f"Encontradas {len(all_images)} imagens. Processando {len(images_to_process)}.")

    # Processa as imagens sequencialmente para evitar bloqueio do banco de dados
    for image_path in images_to_process:
        await process_image(image_path)

    print("\nProcessamento concluído!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classificador de Placas e executor de OCR.")
    parser.add_argument("folder", help="Caminho para a pasta com as imagens a serem processadas.")
    parser.add_argument("--limit", type=int, help="Número máximo de imagens a serem processadas.")
    args = parser.parse_args()
    
    # Inicia o loop de eventos assíncronos
    asyncio.run(main(args.folder, args.limit))
