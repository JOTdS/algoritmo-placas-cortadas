import aiosqlite
import datetime

DB_NAME = "plates.db"

async def init_db():
    """Cria o banco de dados e a tabela se não existirem."""
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                classification TEXT NOT NULL,
                ocr_text TEXT,
                timestamp DATETIME NOT NULL
            )
        """)
        await db.commit()

async def add_record(image_path: str, classification: str, ocr_text: str = None):
    """Adiciona um novo registro de imagem no banco de dados."""
    async with aiosqlite.connect(DB_NAME) as db:
        timestamp = datetime.datetime.now()
        await db.execute("""
            INSERT INTO results (image_path, classification, ocr_text, timestamp)
            VALUES (?, ?, ?, ?)
        """, (image_path, classification, ocr_text, timestamp))
        await db.commit()