# test_db.py
import mysql.connector

def get_connection():
    connection = mysql.connector.connect(
        host='localhost',
        user='root',             # MySQL 사용자 이름
        password='12341234',     # MySQL 비밀번호
        database='project_ai'    # 연결할 데이터베이스 이름
    )
    return connection  # <- 반드시 반환

if __name__ == "__main__":
    conn = get_connection()
    if conn:
        print("DB 연결 성공!")
        conn.close()
    else:
        print("DB 연결 실패!")