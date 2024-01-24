import psycopg2
from dotenv import dotenv_values
from typing import Optional


def load_database():
    secrets = dotenv_values(".env")
    return psycopg2.connect(
        host=secrets["DB_HOST"],
        database=secrets["DB_NAME"],
        user=secrets["DB_USERNAME"],
        password=secrets["DB_PASSWORD"],
    )


def fetch_password(username) -> Optional[str]:
    conn = load_database()
    cur = conn.cursor()
    cur.execute("SELECT password FROM users WHERE username = %s", (username,))
    result = cur.fetchone()
    if not result:
        return None
    password = result[0]
    cur.close()
    conn.close()
    return password


def create_account(username, password, email) -> bool:
    conn = load_database()
    cur = conn.cursor()
    cur.execute("SELECT username FROM users WHERE username = %s", (username,))
    result = cur.fetchone()
    if result:
        return False
    cur.execute(
        "INSERT INTO users (username, password, email) VALUES (%s, %s, %s)",
        (username, password, email),
    )
    conn.commit()
    cur.close()
    conn.close()
    return True
