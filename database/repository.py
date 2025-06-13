from config.database import connect_db
from datetime import datetime

def get_pesid(image_name):
    conn = connect_db()
    cursor = conn.cursor()
    query = f"SELECT pesid FROM imagem WHERE caminho_imagem LIKE '%{image_name}%'"
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result[0] if result else None

def insert_suspect(pesid, image_path):
    conn = connect_db()
    cursor = conn.cursor()
    query = "INSERT INTO solicitacao_suspeita (pesid, data, imagem) VALUES (%s, %s, %s)"
    cursor.execute(query, (pesid, datetime.now(), image_path))
    conn.commit()
    cursor.close()
    conn.close()