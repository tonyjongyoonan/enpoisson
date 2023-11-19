import chess.pgn
from stockfish import Stockfish
import chess

stockfish = Stockfish("/opt/homebrew/Cellar/stockfish/16/bin/stockfish", depth=23)
stockfish.set_depth(12)

pgn_file = "../lichess_db_standard_rated_2013-01.pgn"
pgn = open(pgn_file)
game = chess.pgn.read_game(pgn)


moves = game.mainline_moves()

move_list = list(game.mainline_moves())

stockfish.set_position(move_list)

#start one position in since openings are whatever
curr_move = "Black"

tmp_moves = [move_list[0]]

#dataset of tuples ({}, label) where label is True if move was played and False otherwise
dataset = []
for i in range(len(move_list) - 1):
    stockfish.set_position(tmp_moves)

    # feature set 
    top_moves = stockfish.get_top_moves(10)
    for move in top_moves:
        features = {}
        features["elo"] = game.headers[curr_move + "Elo"]
        features["board"] = stockfish.get_board_visual()
        features["move"] = str(move["Move"])
        label = features["move"].lower() == str(move_list[i + 1]).lower()
        dataset.append((features, label))
        if label:
            break

    tmp_moves.append(move_list[i + 1])
    if curr_move == "White":
        curr_move = "Black"
    else:
        curr_move = "White"
    print(dataset)
