from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from passlib.hash import argon2  # for password hashing
from models import *
from chess_engine import ChessEngine
from generate_prompts import get_analysis
import chess
import database

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

white_model_paths = {1500: "multimodalmodel-white-1500.pth"}
black_model_paths = {1500: "multimodalmodel-black-1500.pth"}
chess_engines: dict[tuple[elo_type, bool], ChessEngine] = {
    (1500, chess.WHITE): ChessEngine(white_model_paths[1500]),
    (1500, chess.BLACK): ChessEngine(black_model_paths[1500]),
}


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
    top_k = position.top_k if position.top_k is not None else 3
    elo = position.elo
    return chess_engines[(elo, position.is_white_move)].get_human_move(
        position.fen, position.last_16_moves, top_k=top_k
    )


@app.post("/get-difficulty")
def get_difficulty(position: ChessPosition):
    # at a high level: of the top 5 human moves, how many of them are blunders/good moves?
    # take the probability of bad moves and sum them.
    pass


# x: a numpy array of top 5 stockfish evaluations
# probabilities: a numpy array of probabilities of playing each move
# def get_difficulty(x, probabilities):
#     scaled_x = x/0.9
#     result = 1 / (1 + np.exp(-scaled_x))
#     sum_x = np.sum(result)
#     normalized_result = result / sum_x
#     return np.dot(normalized_result, probabilities)


@app.post("/get-explanation")
def get_explanation(position: ChessExplanation):
    chess_board = chess.Board(position.fen)
    chess_board.push_san(position.move)
    after_fen = chess_board.fen()
    return get_analysis(position.fen, position.move, after_fen, position.is_white_move)


if __name__ == "__main__":
    print(argon2.hash("password"))
    print(
        argon2.verify(
            "password",
            "$argon2id$v=19$m=65536,t=3,p=4$HUPI2VuLEWKMcW6tVaq1Fg$B1F198SSajgfutyAVl75E0iOYrtB7WTsWpt69A6WnY4",
        )
    )
