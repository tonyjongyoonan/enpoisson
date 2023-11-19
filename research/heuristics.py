import chess.pgn
from stockfish import Stockfish
import chess

def is_passed_pawn_black(board, row, column): 
    for i in range(row-1, -1, -1):
        if (board.piece_at(i*8+(column-1)) != None):
            if (board.piece_at(i*8+(column-1)).symbol() == "P"):
                return False
    for i in range(row-1, -1, -1):
        if (board.piece_at(i*8+(column+1)) != None):
            if (board.piece_at(i*8+(column+1)).symbol() == "P"):
                return False
    return True

def is_passed_pawn_white(board, row, column):
    # returns True if SQUARE is a passed pawn for COLOR
    for i in range(row+1, 8):
        if (board.piece_at(i*8+(column-1)) != None):
            if (board.piece_at(i*8+(column-1)).symbol() == "p"):
                return False
    for i in range(row+1, 8):
        if (board.piece_at(i*8+(column+1)) != None):
            if (board.piece_at(i*8+(column+1)).symbol() == "p"):
                return False
    return True


def count_passed_pawn(board, color):
    # returns number of passed pawns for COLOR
    for i in range(8):
        # count number of pawns in column i
        passed_pawns = 0
        for j in range(8):
            if (board.piece_at(i*8+j) != None and color == chess.BLACK):
                if (board.piece_at(i*8+j).symbol() == "p"):
                    print(i, " ", j)
                    # TODO: fix last argument.
                    if (is_passed_pawn_black(board, i, j)):
                        passed_pawns += 1
            elif (board.piece_at(i*8+j) != None and color == chess.WHITE):
                if (board.piece_at(i*8+j).symbol() == "P"):
                    print(i, " ", j)
                    if (is_passed_pawn_white(board, i, j)):
                        passed_pawns += 1
    return passed_pawns



def bishop_attack(board, square, color):
    # returns True if SQUARE is attacked by opposing color bishop
    # NOTE: this function does not check if the square is occupied by a piece
    # TODO: actually write this. Need to be careful because bishop can be obstructed by another piece
    # may be worthwhile to have an x-ray bishop function
    return

def knight_attack(board, square, color):
    # returns True if SQUARE is attacked by opposing color knight
    # NOTE: this function does not check if the square is occupied by a piece
    if (color == chess.WHITE):
        if (board.piece_at(square+17) != None):
            if (board.piece_at(square+17).symbol() == "n"):
                return True
        if (board.piece_at(square+15) != None):
            if (board.piece_at(square+15).symbol() == "n"):
                return True
        if (board.piece_at(square+10) != None):
            if (board.piece_at(square+10).symbol() == "n"):
                return True
        if (board.piece_at(square+6) != None):
            if (board.piece_at(square+6).symbol() == "n"):
                return True
        if (board.piece_at(square-17) != None):
            if (board.piece_at(square-17).symbol() == "n"):
                return True
        if (board.piece_at(square-15) != None):
            if (board.piece_at(square-15).symbol() == "n"):
                return True
        if (board.piece_at(square-10) != None):
            if (board.piece_at(square-10).symbol() == "n"):
                return True
        if (board.piece_at(square-6) != None):
            if (board.piece_at(square-6).symbol() == "n"):
                return True
    else:
        if (board.piece_at(square+17) != None):
            if (board.piece_at(square+17).symbol() == "N"):
                return True
        if (board.piece_at(square+15) != None):
            if (board.piece_at(square+15).symbol() == "N"):
                return True
        if (board.piece_at(square+10) != None):
            if (board.piece_at(square+10).symbol() == "N"):
                return True
        if (board.piece_at(square+6) != None):
            if (board.piece_at(square+6).symbol() == "N"):
                return True
        if (board.piece_at(square-17) != None):
            if (board.piece_at(square-17).symbol() == "N"):
                return True
        if (board.piece_at(square-15) != None):
            if (board.piece_at(square-15).symbol() == "N"):
                return True
        if (board.piece_at(square-10) != None):
            if (board.piece_at(square-10).symbol() == "N"):
                return True
        if (board.piece_at(square-6) != None):
            if (board.piece_at(square-6).symbol() == "N"):
                return True
    return False


def pawn_attack(board, square, color):
    # returns True if SQUARE is attacked by opposing color pawn
    # NOTE: this function does not check if the square is occupied by a piece
    if (color == chess.WHITE):
        if (board.piece_at(square+9) != None):
            if (board.piece_at(square+9).symbol() == "p"):
                return True
        if (board.piece_at(square+7) != None):
            if (board.piece_at(square+7).symbol() == "p"):
                return True
    else:
        if (board.piece_at(square-9) != None):
            if (board.piece_at(square-9).symbol() == "P"):
                return True
        if (board.piece_at(square-7) != None):
            if (board.piece_at(square-7).symbol() == "P"):
                return True
    return False


def count_black_double_pawns(board):
    double_pawns_count = 0
    for i in range(8):
        # count number of pawns in column i
        pawns_in_column = 0
        for j in range(8):
            if (board.piece_at(i+j*8) != None):
                if (board.piece_at(i+j*8).symbol() == "p"):
                    pawns_in_column += 1
        if (pawns_in_column >= 2):
            double_pawns_count += 1
    return double_pawns_count

def count_white_double_pawns(board):
    double_pawns_count = 0
    for i in range(8):
        # count number of pawns in column i
        pawns_in_column = 0
        for j in range(8):
            if (board.piece_at(i+j*8) != None):
                if (board.piece_at(i+j*8).symbol() == "P"):
                    pawns_in_column += 1
        if (pawns_in_column >= 2):
            double_pawns_count += 1
    return double_pawns_count

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
# pgn = open("../../lichess_db_standard_rated_2017-02.pgn")
pgn = open("single_game.pgn")
game = chess.pgn.read_game(pgn)

move_list = list(game.mainline_moves())
stockfish.set_position(move_list)
board = chess.Board()
for i in move_list:
    board.push(i)
print(board)
print(stockfish.get_board_visual())
print(material_count(board))
print(white_has_bishop_pair(board))
print(black_has_bishop_pair(board))
print(king_pawn_distance(board))
print(count_white_double_pawns(board))
print(count_black_double_pawns(board))
print(pawn_attack(board, chess.G8, chess.BLACK))
print(knight_attack(board, chess.G5, chess.WHITE))
print(count_passed_pawn(board, chess.WHITE))
# print(material_count(stockfish.board()))
