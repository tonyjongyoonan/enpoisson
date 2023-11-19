import chess.pgn
from stockfish import Stockfish
import chess

ruylopez = Stockfish("/opt/homebrew/Cellar/stockfish/16/bin/stockfish", depth=23)
ruylopez.set_depth(23)

moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5c6", "d7c6", "f3e5"]
ruylopez.set_position(moves)
print(ruylopez.get_board_visual())



stockfish2 = Stockfish("/opt/homebrew/Cellar/stockfish/16/bin/stockfish", depth=23)

pgn = open("lichess_db_standard_rated_2017-02.pgn")
# pgn = open("single_game.pgn")
game = chess.pgn.read_game(pgn)

for i in range(8000):
    game = chess.pgn.read_game(pgn)
    move_list = list(game.mainline_moves())
    tmp_list = []
    for move in move_list:
        stockfish2.set_position(tmp_list)
        tmp_list.append(move)
        if (ruylopez.get_board_visual() == stockfish2.get_board_visual()):
            print(i)
            with open("ruy_lopezes.txt", "a") as f:
                f.write(str(list(game.mainline_moves())))


                
