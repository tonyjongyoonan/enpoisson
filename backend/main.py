from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import dotenv_values
import psycopg2
import os


# load database
def load_database():
    secrets = dotenv_values(".env")
    psycopg2.connect(
        host=secrets["DB_HOST"],
        database=secrets["DB_NAME"],
        user=secrets["DB_USERNAME"],
        password=secrets["DB_PASSWORD"],
    )


connection = load_database()

app = FastAPI()
users = {}


class UserCreate(BaseModel):
    username: str
    password: str
    email: str


class UserLogin(BaseModel):
    username: str
    password: str


@app.post("/login")
def validate_login(user: UserLogin):
    # FIXME: this we are essentially storing passwords in plain text; use a salt, and a hash
    # or sign in with Google
    username = user.username
    password = user.password
    if users[username] and users[username].password == password:
        return {"message": f"OK"}
    else:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/account")
def create_account(user: UserCreate):
    if (
        user.username is not None
        and user.password is not None
        and user.email is not None
        and user.username != ""
        and user.username not in users
    ):
        username = user.username
        users[username] = user
        return {"message": f"Succesfully created account for {username}."}
    else:
        raise HTTPException(status_code=409, detail="Bad request.")
