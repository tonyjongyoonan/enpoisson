from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from passlib.hash import argon2  # for password hashing
from models import UserCreate, UserLogin, ChessPosition
from chess_engine import ChessEngine
import database

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

chess_engine_model_path = "multimodalmodel-exp-12.pth"
chess_engine = ChessEngine(chess_engine_model_path)


@app.post("/login")
async def validate_login(user: UserLogin):
    password_check = await database.fetch_password(user.username)
    if password_check is None:
        raise HTTPException(status_code=404, detail="Account not found")
    if argon2.verify(user.password, password_check):
        return {"message": f"OK"}
    else:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/account")
async def create_account(user: UserCreate):
    print(user)
    if (
        user.username is not None
        and user.password is not None
        and user.email is not None
        and user.username != ""
    ):
        hashed_password = argon2.hash(user.password)
        create_success = await database.create_account(
            user.username, hashed_password, user.email
        )
        if not create_success:
            raise HTTPException(status_code=409, detail="Bad request.")
        else:
            return {"message": f"Succesfully created account for {user.username}."}
    else:
        raise HTTPException(status_code=400, detail="Invalid username or password.")


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/get-human-move")
def get_human_move(position: ChessPosition):
    return chess_engine.get_human_move(position.fen, position.last_16_moves, top_k=3)


@app.post("/get-difficulty")
def get_difficulty(position: ChessPosition):
    pass


if __name__ == "__main__":
    print(argon2.hash("password"))
    print(
        argon2.verify(
            "password",
            "$argon2id$v=19$m=65536,t=3,p=4$HUPI2VuLEWKMcW6tVaq1Fg$B1F198SSajgfutyAVl75E0iOYrtB7WTsWpt69A6WnY4",
        )
    )
