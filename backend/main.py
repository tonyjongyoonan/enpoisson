from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from passlib.hash import argon2  # for password hashing
from models import *
from chess_engine import ChessEngine
from generate_prompts import get_analysis
import chess
import database
from stockfish import Stockfish
import numpy as np
import platform

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

supported_configs = [
    (1100, chess.WHITE),
    (1100, chess.BLACK),
    (1500, chess.WHITE),
    (1500, chess.BLACK),
    (2100, chess.WHITE),
    (2100, chess.BLACK),
]


def get_stockfish_path():
    os_type = platform.system().lower()
    os_arch = platform.machine().lower()
    os_release = platform.release().lower()
    if os_type == "linux":
        if "wsl" in os_release:
            return "../stockfish_linux_wsl"
        else:
            return "../stockfish_linux"
    elif os_type == "darwin":
        if "arm" in os_arch:
            return "../stockfish_mac_arm"
        else:
            return "../stockfish_mac_x86"
    raise Exception("Unsupported OS")


stockfish = Stockfish(get_stockfish_path(), depth=8)


def config_to_str(config: tuple[int, bool]) -> str:
    elo, is_white = config
    return f"{'white' if is_white else 'black'}-{elo}"


model_paths = {
    elo_bw: f"multimodalmodel-{config_to_str(elo_bw)}.pth"
    for elo_bw in supported_configs
}
vocab_paths = {
    elo_bw: f"vocab-{config_to_str(elo_bw)}.pkl" for elo_bw in supported_configs
}
chess_engines: dict[tuple[elo_type, bool], ChessEngine] = {
    elo_bw: ChessEngine(model_paths[elo_bw], vocab_paths[elo_bw])
    for elo_bw in supported_configs
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
    """
    Output format:
    { "e4": 0.5 }
    """
    top_k = position.top_k if position.top_k is not None else 3
    elo = position.elo
    return chess_engines[(elo, position.is_white_move)].get_human_move(
        position.fen, position.last_16_moves, top_k=top_k
    )


@app.post("/get-difficulty")
def get_difficulty(position: Difficulty):
    # at a high level: of the top 5 human moves, how many of them are blunders/good moves?
    # take the probability of bad moves and sum them.
    stockfish.set_fen_position(position.fen)
    chess_board = chess.Board(position.fen)
    # each move is a dictionary with keys "Move", "Centipawn", "Mate"
    stockfish_top_5 = stockfish.get_top_moves(5)
    eval = stockfish.get_evaluation()
    if eval["type"] == "mate":
        num_moves = eval["value"]
        eval_str = f'{"" if num_moves > 0 else "-"}M{abs(num_moves)}'
    else:
        eval_str = f"{eval['value'] / 100.0:.2f}"

    stockfish_top_5_not_none = [
        move for move in stockfish_top_5 if move["Centipawn"] is not None
    ]
    turn = chess.WHITE if position.is_white_move else chess.BLACK

    moves_uci = [move["Move"] for move in stockfish_top_5_not_none]
    evals = [int(move["Centipawn"]) / 100.0 for move in stockfish_top_5_not_none]
    moves_san = [chess_board.san(chess.Move.from_uci(move)) for move in moves_uci]
    engine = chess_engines[(position.elo, position.is_white_move)]
    probabilities = [
        engine.get_probability_of_move(position.fen, position.last_16_moves, move)
        for move in moves_san
    ]
    probabilities_zeroed = np.array([x if x is not None else 0 for x in probabilities])
    scaled_evals = np.array(evals) / 0.9
    result = 1 / (1 + np.exp(-scaled_evals))
    if turn == chess.BLACK:
        result = 1 - result
    sum_x = np.sum(result)
    normalized_result = result / sum_x
    return {
        "eval": eval_str,
        "difficulty": np.dot(normalized_result, probabilities_zeroed),
        "stockfish_move": moves_san[0],
    }


@app.post("/get-explanation")
def get_explanation(position: ChessExplanation):
    chess_board = chess.Board(position.fen)
    chess_board.push_san(position.move)
    after_fen = chess_board.fen()
    return get_analysis(position.fen, position.move, after_fen, position.is_white_move)


if __name__ == "__main__":
    # print(argon2.hash("password"))
    # print(
    #     argon2.verify(
    #         "password",
    #         "$argon2id$v=19$m=65536,t=3,p=4$HUPI2VuLEWKMcW6tVaq1Fg$B1F198SSajgfutyAVl75E0iOYrtB7WTsWpt69A6WnY4",
    #     )
    # )
    print(
        get_difficulty(
            Difficulty(
                fen="rnbqkbnr/ppp2pp1/8/3pp2p/4P3/4K3/PPPP1PPP/RNBQ1BNR w kq - 0 4",
                last_16_moves=["e4", "e5", "Ke2", "h5", "Ke3", "d5"],
                is_white_move=True,
            )
        )
    )
