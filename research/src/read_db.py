import chess.pgn
from stockfish import Stockfish
import chess

stockfish = Stockfish("/opt/homebrew/Cellar/stockfish/16/bin/stockfish", depth=23)
stockfish.set_depth(23)

pgn_file = "lichess_db_standard_rated_2017-02.pgn"
pgn = open(pgn_file)
game = chess.pgn.read_game(pgn)


# print(game.headers)
# print(game)
# game = chess.pgn.read_game(pgn)
# print moves of game
moves = game.mainline_moves()

# for move in moves:
#     print(move)
move_list = list(game.mainline_moves())
# setup = ["e4"]
stockfish.set_position(move_list)

tmp_list = []
for move in move_list:
    stockfish.set_position(tmp_list)
    tmp_list.append(move)
    print(stockfish.get_board_visual())
