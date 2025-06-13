import mysql.connector

DB_CONFIG = {
    'user': 'root',
    'password': '040498',
    'host': '127.0.0.1',
    'database': 'face_recognition',
    'port': 3306
}

def connect_db():
    return mysql.connector.connect(**DB_CONFIG)
