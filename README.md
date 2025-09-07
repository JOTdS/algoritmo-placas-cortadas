# Algoritmo de Classificação e OCR de Placas

Este projeto utiliza uma Rede Neural Convolucional (CNN) para classificar imagens de placas de veículos como 'válidas' ou 'inválidas'. Para as placas classificadas como 'válidas', o sistema executa um Reconhecimento Óptico de Caracteres (OCR) para extrair o texto da placa. Todos os resultados são armazenados em um banco de dados SQLite.

## Como Funciona

O projeto é dividido em três módulos principais:

1.  **`model.py` - Treinamento do Modelo:**
    - Este script constrói uma CNN usando a biblioteca TensorFlow/Keras.
    - Ele treina o modelo com base nas imagens fornecidas no diretório `dataset/`, que deve conter as subpastas `validos/` e `invalidos/`.
    - Após o treinamento, o modelo é salvo no arquivo `plate_classifier.keras`.

2.  **`database.py` - Gerenciamento do Banco de Dados:**
    - Define a estrutura do banco de dados SQLite (`plates.db`).
    - Contém funções assíncronas para inicializar o banco e para adicionar os registros dos resultados da classificação e do OCR.

3.  **`main.py` - Orquestração Principal:**
    - Carrega o modelo treinado (`plate_classifier.keras`).
    - Aceita um caminho para uma pasta de imagens e uma flag opcional `--limit` como argumentos de linha de comando.
    - Itera recursivamente sobre todas as imagens `.jpg` na pasta e suas subpastas.
    - Para cada imagem (respeitando o limite, se fornecido), prevê a classe ('valido' ou 'invalido').
    - Se a imagem for 'valida', executa o OCR com Pytesseract.
    - Salva o caminho da imagem, a classificação, o texto do OCR (se aplicável) e um timestamp no banco de dados `plates.db`.

## Pré-requisitos

Para executar este projeto, você precisará ter o seguinte software instalado:

1.  **Python 3.x**
2.  **Pip** (gerenciador de pacotes Python)
3.  **Tesseract OCR Engine:** Este é o motor de OCR usado pelo Pytesseract.
    - Em sistemas Debian/Ubuntu, instale com:
      ```bash
      sudo apt-get install tesseract-ocr
      ```
    - Para outros sistemas (Windows, macOS), consulte a [documentação oficial do Tesseract](https://github.com/tesseract-ocr/tesseract).

## Como Executar

1.  **Instale as dependências Python:**
    Navegue até a pasta do projeto e execute o seguinte comando para instalar as bibliotecas necessárias:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Treine o Modelo de Classificação:**
    Para treinar o modelo com as imagens do diretório `dataset/`, execute:
    ```bash
    python model.py
    ```
    *Este passo só é necessário na primeira vez ou se você modificar o conjunto de dados de treinamento.*

3.  **Execute a Classificação e o OCR:**
    Para processar um diretório de imagens, execute o script `main.py` seguido pelo caminho da pasta. Você pode usar a flag opcional `--limit` para definir um número máximo de imagens a serem processadas.
    ```bash
    python main.py /caminho/para/sua/pasta_de_imagens/ [--limit N]
    ```

## Exemplos de Uso

- **Processar as imagens de exemplo da pasta `dataset/invalidos`:**

  ```bash
  python main.py dataset/invalidos/
  ```

- **Processar uma pasta de imagens localizada em `/home/usuario/novas_placas/`:**

  ```bash
  python main.py /home/usuario/novas_placas/
  ```

- **Processar apenas as 10 primeiras imagens de um diretório:**

  ```bash
  python main.py dataset/ --limit 10
  ```

Após a execução, os resultados podem ser consultados no arquivo `plates.db` usando qualquer ferramenta de visualização de SQLite.
>>>>>>> 5baab81 (Inicializando projeto)
