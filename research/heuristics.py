import chess.pgn
from stockfish import Stockfish
import chess


def king_pawn_distance(board):
    # find minimum distance between white king and white pawns
    white_king_square = board.king(chess.WHITE)
    white_pawn_squares = board.pieces(chess.PAWN, chess.WHITE)
    white_pawn_distance = 8
    for i in white_pawn_squares:
        white_pawn_distance = min(white_pawn_distance, chess.square_distance(i, white_king_square))
    # find minimum distance between black king and black pawns
    black_king_square = board.king(chess.BLACK)
    black_pawn_squares = board.pieces(chess.PAWN, chess.BLACK)
    black_pawn_distance = 8
    for i in black_pawn_squares:
        black_pawn_distance = min(black_pawn_distance, chess.square_distance(i, black_king_square))
    return white_pawn_distance, black_pawn_distance


def black_has_bishop_pair(board):
    black_bishops = 0
    for i in range(8):
        for j in range(8):
            if (board.piece_at(i*8+j) != None):
                if (board.piece_at(i*8+j).symbol() == "b"):
                    black_bishops += 1
    if (black_bishops >= 2):
        return True
    else:
        return False


def white_has_bishop_pair(board):
    white_bishops = 0
    for i in range(8):
        for j in range(8):
            if (board.piece_at(i*8+j) != None):
                if (board.piece_at(i*8+j).symbol() == "B"):
                    white_bishops += 1
    if (white_bishops >= 2):
        return True
    else:
        return False


def material_count(board):
    # stockfish endgame piece values
    piece_value_endgame = {
        "p": 206,
        "n": 854,
        "b": 915,
        "r": 1380,
        "q": 2682,
        "k": 0
    } 
    letters_lower = "pnbrq"
    material_count_white = 0
    material_count_black = 0
    for i in range(8):
        for j in range(8):
           # goes from a1 to a8 to h8
           if (board.piece_at(i*8+j) != None):
               if (board.piece_at(i*8+j).symbol() in letters_lower):
                   material_count_black += piece_value_endgame[board.piece_at(i*8+j).symbol()]
               else:
                   material_count_white += piece_value_endgame[board.piece_at(i*8+j).symbol().lower()]
    return material_count_white, material_count_black


stockfish = Stockfish("/opt/homebrew/Cellar/stockfish/16/bin/stockfish", depth=23)
# read first game in ruy_lopezes.txt
pgn = open("lichess_db_standard_rated_2017-02.pgn")
game = chess.pgn.read_game(pgn)
move_list = list(game.mainline_moves())
stockfish.set_position(move_list)
board = chess.Board()
for i in move_list:
    board.push(i)
print(board)
print(stockfish.get_board_visual())
print(material_count(board))
print(king_pawn_distance(board))
# print(material_count(stockfish.board()))
