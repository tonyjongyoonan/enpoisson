import chess.pgn
from stockfish import Stockfish
import chess
import time

def get_column(square):
    return square % 8

def get_column_a(i, j):
    return j

def get_row(square):
    return square // 8

def get_row_a(i, j):
    return i

def is_passed_pawn_black(board, row, column): 
    for i in range(row-1, -1, -1):
        if (board.piece_at(i*8 + column) != None):
            if (board.piece_at(i*8 + column).symbol() == "P"):
                return False
    for i in range(row-1, -1, -1):
        if (column >= 1):
            if (board.piece_at(i*8+(column-1)) != None):
                if (board.piece_at(i*8+(column-1)).symbol() == "P"):
                    return False
    for i in range(row-1, -1, -1):
        if (column <= 6):
            if (board.piece_at(i*8+(column+1)) != None):
                if (board.piece_at(i*8+(column+1)).symbol() == "P"):
                    return False
    return True

def is_passed_pawn_white(board, row, column):
    # returns True if SQUARE is a passed pawn for COLOR
    for i in range(row+1, 8):
        if (board.piece_at(i*8+column) != None):
            if (board.piece_at(i*8+column).symbol() == "p"):
                return False
    for i in range(row+1, 8):
        if (column >= 1):
            if (board.piece_at(i*8+(column-1)) != None):
                if (board.piece_at(i*8+(column-1)).symbol() == "p"):
                    return False
    for i in range(row+1, 8):
        if (column <= 6):
            if (board.piece_at(i*8+(column+1)) != None):
                if (board.piece_at(i*8+(column+1)).symbol() == "p"):
                    return False
    return True


def count_passed_pawn(board, color):
    # returns number of passed pawns for COLOR
    passed_pawns = 0
    for i in range(8):
        # count number of pawns in column i
        for j in range(8):
            if (board.piece_at(i*8+j) != None and color == chess.BLACK):
                if (board.piece_at(i*8+j).symbol() == "p"):
                    if (is_passed_pawn_black(board, i, j)):
                        passed_pawns += 1
            elif (board.piece_at(i*8+j) != None and color == chess.WHITE):
                if (board.piece_at(i*8+j).symbol() == "P"):
                    if (is_passed_pawn_white(board, i, j)):
                        passed_pawns += 1
    return passed_pawns


def attacks(board, square, color): 
    # returns bitboard of pieces of COLOR that attack SQUARE
    return board.attackers(color, square)



def bishop_attack(board, square, color):
    # returns True if SQUARE is attacked by opposing color bishop
    # NOTE: this function does not check if the square is occupied by a piece
    # TODO: actually write this. Need to be careful because bishop can be obstructed by another piece
    # may be worthwhile to have an x-ray bishop function
    return

def knight_attack(board, square, color):
    # returns True if SQUARE is attacked by opposing color knight
    # NOTE: this function does not check if the square is occupied by a piece
    piece = "n" if color == chess.WHITE else "N"

    if (get_column(square) <= 6 and get_row(square) <= 5 and board.piece_at(square+17) != None):
        if (board.piece_at(square+17).symbol() == piece):
            return True
    if (get_column(square) >= 1 and get_row(square) <= 5 and board.piece_at(square+15) != None):
        if (board.piece_at(square+15).symbol() == piece):
            return True
    if (get_column(square) <= 5 and get_row(square) <= 6 and board.piece_at(square+10) != None):
        if (board.piece_at(square+10).symbol() == piece):
            return True
    if (get_column(square) >= 1 and get_row(square) <= 6 and board.piece_at(square+6) != None):
        if (board.piece_at(square+6).symbol() == piece):
            return True
    if (get_column(square) >= 1 and get_row(square) >= 2 and board.piece_at(square-17) != None):
        if (board.piece_at(square-17).symbol() == piece):
            return True
    if (get_column(square) <= 6 and get_row(square) >= 2 and board.piece_at(square-15) != None):
        if (board.piece_at(square-15).symbol() == piece):
            return True
    if (get_column(square) >= 2 and get_row(square) >= 1 and board.piece_at(square-10) != None):
        if (board.piece_at(square-10).symbol() == piece):
            return True
    if (get_column(square) <= 5 and get_row(square) >= 1 and board.piece_at(square-6) != None):
        if (board.piece_at(square-6).symbol() == piece):
            return True
    return False

def king_attack(board, square, color):
    # returns True if SQUARE is attacked by opposing color king
    # NOTE: this function does not check if the square is occupied by a piece
    piece = "k" if color == chess.WHITE else "K"

    if (get_column(square) <= 6 and board.piece_at(square+1) != None):
        if (board.piece_at(square+1).symbol() == piece):
            return True
    if (get_column(square) >= 1 and board.piece_at(square-1) != None):
        if (board.piece_at(square-1).symbol() == piece):
            return True
    if (get_row(square) <= 6 and board.piece_at(square+8) != None):
        if (board.piece_at(square+8).symbol() == piece):
            return True
    if (get_row(square) >= 1 and board.piece_at(square-8) != None):
        if (board.piece_at(square-8).symbol() == piece):
            return True
    if (get_column(square) <= 6 and get_row(square) <= 6 and board.piece_at(square+9) != None):
        if (board.piece_at(square+9).symbol() == piece):
            return True
    if (get_column(square) >= 1 and get_row(square) <= 6 and board.piece_at(square+7) != None):
        if (board.piece_at(square+7).symbol() == piece):
            return True
    if (get_column(square) >= 1 and get_row(square) >= 1 and board.piece_at(square-9) != None):
        if (board.piece_at(square-9).symbol() == piece):
            return True
    if (get_column(square) <= 6 and get_row(square) >= 1 and board.piece_at(square-7) != None):
        if (board.piece_at(square-7).symbol() == piece):
            return True
    return False


def pawn_attack(board, square, color):
    # returns True if SQUARE is attacked by opposing color pawn
    # NOTE: this function does not check if the square is occupied by a piece

    if (color == chess.WHITE):
        if (get_column(square) <= 6 and get_row(square) <= 6 and board.piece_at(square+9) != None):
            if (board.piece_at(square+9).symbol() == "p"):
                print("REEEEEEEEE")
                return True
        if (get_column(square) >= 1 and get_row(square) <= 6 and board.piece_at(square+7) != None):
            if (board.piece_at(square+7).symbol() == "p"):
                print("DHFSPDFPHFSDFSDF")
                return True
    else:
        if (get_column(square) >= 1 and get_row(square) >= 1 and board.piece_at(square-9) != None):
            if (board.piece_at(square-9).symbol() == "P"):
                return True
        if (get_column(square) <= 6 and get_row(square) >= 1 and board.piece_at(square-7) != None):
            if (board.piece_at(square-7).symbol() == "P"):
                return True
    return False

def knight_control(board, square, color):
    # NOTE: can enhance by ignoring certain squares if they are occupied by friendly pieces
    # for example, stockfish does not add 1 if the square is occupied by the color's queen
    piece = "N" if color == chess.WHITE else "n"

    count = 0
    if (get_column(square) <= 6 and get_row(square) <= 5 and board.piece_at(square+17) != None):
        if (board.piece_at(square+17).symbol() == piece):
            count += 1
    if (get_column(square) >= 1 and get_row(square) <= 5 and board.piece_at(square+15) != None):
        if (board.piece_at(square+15).symbol() == piece):
            count += 1
    if (get_column(square) <= 5 and get_row(square) <= 6 and board.piece_at(square+10) != None):
        if (board.piece_at(square+10).symbol() == piece):
            count += 1
    if (get_column(square) >= 1 and get_row(square) <= 6 and board.piece_at(square+6) != None):
        if (board.piece_at(square+6).symbol() == piece):
            count += 1
    if (get_column(square) >= 1 and get_row(square) >= 2 and board.piece_at(square-17) != None):
        if (board.piece_at(square-17).symbol() == piece):
            count += 1
    if (get_column(square) <= 6 and get_row(square) >= 2 and board.piece_at(square-15) != None):
        if (board.piece_at(square-15).symbol() == piece):
            count += 1
    if (get_column(square) >= 2 and get_row(square) >= 1 and board.piece_at(square-10) != None):
        if (board.piece_at(square-10).symbol() == piece):
            count += 1
    if (get_column(square) <= 5 and get_row(square) >= 1 and board.piece_at(square-6) != None):
        if (board.piece_at(square-6).symbol() == piece):
            count += 1
    return count

def pawn_control(board, square, color):
    # returns True if SQUARE is controlled by opposing color pawn
    # NOTE: this function does not check if the square is occupied by a piece
    count = 0
    if (color == chess.BLACK):
        if (get_column(square) <= 6 and get_row(square) <= 6 and board.piece_at(square+9) != None):
            if (board.piece_at(square+9).symbol() == "p"):
                count += 1
        if (get_column(square) >= 1 and get_row(square) <= 6 and board.piece_at(square+7) != None):
            if (board.piece_at(square+7).symbol() == "p"):
                count += 1
    else:
        if (get_column(square) >= 1 and get_row(square) >= 1 and board.piece_at(square-9) != None):
            if (board.piece_at(square-9).symbol() == "P"):
                count += 1
        if (get_column(square) <= 6 and get_row(square) >= 1 and board.piece_at(square-7) != None):
            if (board.piece_at(square-7).symbol() == "P"):
                count += 1
    return count

def king_control(board, square, color):
    # returns True if SQUARE is controlled by opposing color king
    # NOTE: this function does not check if the square is occupied by a piece
    piece = "K" if color == chess.WHITE else "k"

    if (get_column(square) <= 6 and board.piece_at(square+1) != None):
        if (board.piece_at(square+1).symbol() == piece):
            return 1
    if (get_column(square) >= 1 and board.piece_at(square-1) != None):
        if (board.piece_at(square-1).symbol() == piece):
            return 1
    if (get_row(square) <= 6 and board.piece_at(square+8) != None):
        if (board.piece_at(square+8).symbol() == piece):
            return 1
    if (get_row(square) >= 1 and board.piece_at(square-8) != None):
        if (board.piece_at(square-8).symbol() == piece):
            return 1
    if (get_column(square) <= 6 and get_row(square) <= 6 and board.piece_at(square+9) != None):
        if (board.piece_at(square+9).symbol() == piece):
            return 1
    if (get_column(square) >= 1 and get_row(square) <= 6 and board.piece_at(square+7) != None):
        if (board.piece_at(square+7).symbol() == piece):
            return 1
    if (get_column(square) >= 1 and get_row(square) >= 1 and board.piece_at(square-9) != None):
        if (board.piece_at(square-9).symbol() == piece):
            return 1
    if (get_column(square) <= 6 and get_row(square) >= 1 and board.piece_at(square-7) != None):
        if (board.piece_at(square-7).symbol() == piece):
            return 1
    return 0

#TODO: Check if this function actually works as intended
def bishop_xray_control(board, square, color):
    # returns True if SQUARE is controlled by opposing color bishop
    symbol = "B" if color == chess.WHITE else "b"
    v = 0
    for i in range(4):
        ix = ((i > 1) * 2) - 1
        iy = ((i %2 == 0) * 2) - 1
        for d in range(1, 8):
            if (get_column(square) + d * ix <= 7) and \
                        (get_column(square) + d * ix >= 0) and \
                        (get_row(square) + d * iy <= 7) and \
                        (get_row(square) + d * iy >= 0):
                b = board.piece_at(square + d * ix + d * iy * 8)
                if b is not None:
                    if b.symbol() == symbol:
                        dir = pinned_direction(board, square + d * ix + d * iy * 8)
                        if (dir == 0 or abs(ix + iy * 3) == dir):
                            v += 1
                if b is not None and b.symbol() != "Q" and b.symbol() != "q":
                    break
    return v

def rook_xray_control(board, square, color):
    symbol = "R" if color == chess.WHITE else "r"
    v = 0
    for i in range(4):
        ix = 0
        if i == 0:
            ix = -1
        elif i == 1:
            ix = 1
        iy = 0
        if i == 2:
            iy = -1
        elif i == 3:
            iy = 1
        for d in range(1, 8):
            if (get_column(square) + d * ix <= 7) and \
                    (get_column(square) + d * ix >= 0) and \
                    (get_row(square) + d * iy <= 7) and \
                    (get_row(square) + d * iy >= 0):
                b = board.piece_at(square + d * ix + d * iy * 8)
                if b is not None:
                    if b.symbol() == symbol:
                        dir = pinned_direction(board, square + d * ix + d * iy * 8)
                        if (dir == 0 or abs(ix + iy * 3) == dir):
                            v += 1
                if b is not None and b.symbol() != "Q" and b.symbol() != "q" and b.symbol() != symbol:
                    break
    return v

def queen_control(board, square, color):
    v = 0
    for i in range(8):
        ix = (i + (i > 3)) % 3 - 1
        iy = (((i + (i > 3)) // 3)) - 1
        for d in range(1, 8):
            if (get_column(square) + d * ix <= 7) and \
                    (get_column(square) + d * ix >= 0) and \
                    (get_row(square) + d * iy <= 7) and \
                    (get_row(square) + d * iy >= 0): 
                b = board.piece_at(square + d * ix + d * iy * 8)
                if b is not None:
                    if b.symbol() == "Q":
                        dir = pinned_direction(board, square + d * ix + d * iy * 8)
                        if (dir == 0 or abs(ix + iy * 3) == dir):
                            v += 1
                if b is not None:
                    break
    return v



def total_control(board, color):
    s = ""
    control = 0
    for i in range(8):
        tmp = ""
        for j in range(8):
            count = 0
            count += knight_control(board, i*8+j, color)
            count += pawn_control(board, i*8+j, color)
            count += king_control(board, i*8+j, color)
            count += bishop_xray_control(board, i*8+j, color)
            count += rook_xray_control(board, i*8+j, color)
            count += queen_control(board, i*8+j, color)
            # print("Total control for ", color, " at square ", i*8+j, " is ", count)
            tmp += str(count) + " "
            control += count
        s = tmp + "\n" + s
    # print(s)
    return control


def pawn_bonus(board, square, color):
    # returns increased pawn score based on rank of pawn.
    # multiplier values chosen arbitrarily
    pawn_value = 206
    multiplier = [0, 1, 1, 1.05, 1.1, 1.3, 2]
    piece = "P" if color == chess.WHITE else "p"
    if (board.piece_at(square) != None and board.piece_at(square).symbol() == piece):
        return pawn_value * multiplier[get_row(square)]
    else:
        return 0

def count_squares_knight_attacks(board, square):
    # returns the number of squares within the chessboard a knight on square SQUARE can attack
    count = 0
    if (get_column(square) <= 6 and get_row(square) <= 5):
        count += 1
    if (get_column(square) >= 1 and get_row(square) <= 5):
        count += 1
    if (get_column(square) <= 5 and get_row(square) <= 6):
        count += 1
    if (get_column(square) >= 1 and get_row(square) <= 6):
        count += 1
    if (get_column(square) >= 1 and get_row(square) >= 2):
        count += 1
    if (get_column(square) <= 6 and get_row(square) >= 2):
        count += 1
    if (get_column(square) >= 2 and get_row(square) >= 1):
        count += 1
    if (get_column(square) <= 5 and get_row(square) >= 1):
        count += 1
    return count

def knight_bonus(board, square, color):
    # knight is more powerful when it is closer to the center.
    multiplier = [0.85, 0.85, 0.85, 1, 1, 1.05, 1.15, 1.2, 1.2]
    piece = "N" if color == chess.WHITE else "n"
    if (board.piece_at(square) != None and board.piece_at(square).symbol() == piece):
        return multiplier[count_squares_knight_attacks(board, square)]
    else:
        return 0
    

def weighted_bonus(board, color):
    # returns weighted bonus for COLOR
    # NOTE: only considers pawns and knights for now.
    score = 0
    for i in range(8):
        for j in range(8):
            score += pawn_bonus(board, i*8+j, color)
            score += knight_bonus(board, i*8+j, color)
    return score


# stealing stockfish code entirely here
# NOTE: idk if this is actually correct
def pinned_direction(board, square):
    color = 1
    if board.piece_at(square).color == chess.BLACK:
        color = -1
    for i in range(8):
        ix = (i + (i > 3)) % 3 - 1
        iy = (i + (i > 3)) // 3 - 1
        king = False
        for d in range(1, 8):
            if (get_column(square) + d * ix <= 7) and (get_column(square) + d * ix >= 0) and \
            (get_row(square) + d * iy <= 7) and (get_row(square) + d * iy >= 0):
                if board.piece_at(square + d * ix + d * iy * 8) is not None:
                    if board.piece_at(square + d * ix + d * iy * 8).symbol() == "K":
                        king = True
                    break
        if king:
            for d in range(1, 8):
                if (get_column(square) - d * ix <= 7) and (get_column(square) - d * ix >= 0) and \
                (get_row(square) - d * iy <= 7) and (get_row(square) - d * iy >= 0):
                    if board.piece_at(square - d * ix - d * iy * 8) is not None:
                        if board.piece_at(square - d * ix - d * iy * 8).symbol() == "Q":
                            return abs(ix + iy * 3) * color
                        elif board.piece_at(square - d * ix - d * iy * 8).symbol() == "B" and (ix * iy != 0):
                            return abs(ix + iy * 3) * color
                        elif board.piece_at(square - d * ix - d * iy * 8).symbol() == "R" and (ix * iy == 0):
                            return abs(ix + iy * 3) * color
                        break

    return 0



            



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


def white_is_sac(board, move):
    # returns True if MOVE is a sacrifice
    # TODO: Actually write this function. completely cheese function rn
    # TODO: make sure defended pieces aren't deemed as sacs LOL
    board.push(move)
    # find square that piece just moved to
    square = move.to_square
    boolea = pawn_attack(board, square, chess.WHITE) or knight_attack(board, square, chess.WHITE) or king_attack(board, square, chess.WHITE)
    board.pop()
    return boolea

def black_is_sac(board, move):
    # returns True if MOVE is a sacrifice
    board.push(move)
    # find square that piece just moved to
    square = move.to_square
    # print("square: ", square)
    boolea = pawn_attack(board, square, chess.BLACK) or knight_attack(board, square, chess.BLACK) or king_attack(board, square, chess.BLACK)
    board.pop()
    return boolea

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


def get_white_material_delta(board, move):
    # returns material delta for white after MOVE
    curr_material = material_count(board)[0]
    board.push(move)
    new_material = material_count(board)[0]
    board.pop()
    return new_material - curr_material

def get_black_material_delta(board, move):
    # returns material delta for black after MOVE
    curr_material = material_count(board)[1]
    board.push(move)
    new_material = material_count(board)[1]
    board.pop()
    return new_material - curr_material

def white_delta_bishop_pair(board, move):
    # returns tuple (x, y) where x, y \in {True, False} where x is True if white has bishop pair before MOVE and y is True if white has bishop pair after MOVE
    curr_bishop_pair = white_has_bishop_pair(board)
    board.push(move)
    new_bishop_pair = white_has_bishop_pair(board)
    board.pop()
    return curr_bishop_pair, new_bishop_pair

def black_delta_bishop_pair(board, move):
    # returns tuple (x, y) where x, y \in {True, False} where x is True if black has bishop pair before MOVE and y is True if black has bishop pair after MOVE
    curr_bishop_pair = black_has_bishop_pair(board)
    board.push(move)
    new_bishop_pair = black_has_bishop_pair(board)
    board.pop()
    return curr_bishop_pair, new_bishop_pair

def white_delta_king_pawn_distance(board, move):
    curr_king_pawn_distance = king_pawn_distance(board)[0]
    board.push(move)
    new_king_pawn_distance = king_pawn_distance(board)[0]
    board.pop()
    return new_king_pawn_distance - curr_king_pawn_distance

def black_delta_king_pawn_distance(board, move):
    curr_king_pawn_distance = king_pawn_distance(board)[1]
    board.push(move)
    new_king_pawn_distance = king_pawn_distance(board)[1]
    board.pop()
    return new_king_pawn_distance - curr_king_pawn_distance

def white_delta_double_pawns(board, move):
    curr_double_pawns = count_white_double_pawns(board)
    board.push(move)
    new_double_pawns = count_white_double_pawns(board)
    board.pop()
    return new_double_pawns - curr_double_pawns

def black_delta_double_pawns(board, move):
    curr_double_pawns = count_black_double_pawns(board)
    board.push(move)
    new_double_pawns = count_black_double_pawns(board)
    board.pop()
    return new_double_pawns - curr_double_pawns

def white_delta_passed_pawns(board, move):
    curr_passed_pawns = count_passed_pawn(board, chess.WHITE)
    board.push(move)
    new_passed_pawns = count_passed_pawn(board, chess.WHITE)
    board.pop()
    return new_passed_pawns - curr_passed_pawns

def black_delta_passed_pawns(board, move):
    curr_passed_pawns = count_passed_pawn(board, chess.BLACK)
    board.push(move)
    new_passed_pawns = count_passed_pawn(board, chess.BLACK)
    board.pop()
    return new_passed_pawns - curr_passed_pawns

def white_delta_total_control(board, move):
    curr_total_control = total_control(board, chess.WHITE)
    board.push(move)
    new_total_control = total_control(board, chess.WHITE)
    board.pop()
    return new_total_control - curr_total_control

def black_delta_total_control(board, move):
    curr_total_control = total_control(board, chess.BLACK)
    board.push(move)
    new_total_control = total_control(board, chess.BLACK)
    board.pop()
    return new_total_control - curr_total_control

def white_delta_weighted_bonus(board, move):
    curr_weighted_bonus = weighted_bonus(board, chess.WHITE)
    board.push(move)
    new_weighted_bonus = weighted_bonus(board, chess.WHITE)
    board.pop()
    return new_weighted_bonus - curr_weighted_bonus

def black_delta_weighted_bonus(board, move):
    curr_weighted_bonus = weighted_bonus(board, chess.BLACK)
    board.push(move)
    new_weighted_bonus = weighted_bonus(board, chess.BLACK)
    board.pop()
    return new_weighted_bonus - curr_weighted_bonus



def get_all_heuristics(board):
    print(board)
    print(stockfish.get_board_visual())
    print("White material count: ", material_count(board)[0])
    print("Black material count: ", material_count(board)[1])
    print("White has bishop pair?: ", white_has_bishop_pair(board))
    print("Black has bishop pair?: ", black_has_bishop_pair(board))
    print("Shortest distance from WHITE king to WHITE pawn: ", king_pawn_distance(board)[0])
    print("Shortest distance from BLACK king to BLACK pawn: ", king_pawn_distance(board)[1])
    print("Number of sets of doubled pawns for white: ", count_white_double_pawns(board))
    print("Number of sets of doubled pawns for black: ", count_black_double_pawns(board))
    print(pawn_attack(board, chess.G8, chess.BLACK))
    print(knight_attack(board, chess.G5, chess.WHITE))
    print("Number of passed pawns for white: ", count_passed_pawn(board, chess.WHITE))
    print("Number of passed pawns for black: ", count_passed_pawn(board, chess.BLACK))
    print("Total control for white: ", total_control(board, chess.WHITE))
    print("Total control for black: ", total_control(board, chess.BLACK))
    print("Weighted bonus for white: ", weighted_bonus(board, chess.WHITE))
    print("Weighted bonus for black: ", weighted_bonus(board, chess.BLACK))


def get_all_delta_for_move(board, move):
    white_material_delta = get_white_material_delta(board, move)
    black_material_delta = get_black_material_delta(board, move)
    white_bishop_pair_delta = white_delta_bishop_pair(board, move)
    black_bishop_pair_delta = black_delta_bishop_pair(board, move)
    white_king_pawn_distance_delta = white_delta_king_pawn_distance(board, move)
    black_king_pawn_distance_delta = black_delta_king_pawn_distance(board, move)
    white_double_pawns_delta = white_delta_double_pawns(board, move)
    black_double_pawns_delta = black_delta_double_pawns(board, move)
    white_passed_pawns_delta = white_delta_passed_pawns(board, move)
    black_passed_pawns_delta = black_delta_passed_pawns(board, move)
    white_total_control_delta = white_delta_total_control(board, move)
    black_total_control_delta = black_delta_total_control(board, move)
    white_weighted_bonus_delta = white_delta_weighted_bonus(board, move)
    black_weighted_bonus_delta = black_delta_weighted_bonus(board, move)
    is_sac_white = white_is_sac(board, move)
    is_sac_black = black_is_sac(board, move)
    print("----MOVE---- : ", move)
    print("White material delta: ", white_material_delta)
    print("Black material delta: ", black_material_delta)
    print("White bishop pair delta: ", white_bishop_pair_delta)
    print("Black bishop pair delta: ", black_bishop_pair_delta)
    print("White king pawn distance delta: ", white_king_pawn_distance_delta)
    print("Black king pawn distance delta: ", black_king_pawn_distance_delta)
    print("White double pawns delta: ", white_double_pawns_delta)
    print("Black double pawns delta: ", black_double_pawns_delta)
    print("White passed pawns delta: ", white_passed_pawns_delta)
    print("Black passed pawns delta: ", black_passed_pawns_delta)
    print("White total control delta: ", white_total_control_delta)
    print("Black total control delta: ", black_total_control_delta)
    print("White weighted bonus delta: ", white_weighted_bonus_delta)
    print("Black weighted bonus delta: ", black_weighted_bonus_delta)
    print("Is white sac: ", is_sac_white)
    print("Is black sac: ", is_sac_black)
    calc_white_difficulty = white_difficulty(board, move, white_material_delta, black_material_delta, white_bishop_pair_delta, black_bishop_pair_delta, white_king_pawn_distance_delta,
                black_king_pawn_distance_delta, white_double_pawns_delta, black_double_pawns_delta, white_passed_pawns_delta, black_passed_pawns_delta,
                white_total_control_delta, black_total_control_delta, white_weighted_bonus_delta, black_weighted_bonus_delta, is_sac_white, is_sac_black)
    calc_black_difficulty = black_difficulty(board, move, white_material_delta, black_material_delta, white_bishop_pair_delta, black_bishop_pair_delta, white_king_pawn_distance_delta,
                black_king_pawn_distance_delta, white_double_pawns_delta, black_double_pawns_delta, white_passed_pawns_delta, black_passed_pawns_delta,
                white_total_control_delta, black_total_control_delta, white_weighted_bonus_delta, black_weighted_bonus_delta, is_sac_white, is_sac_black)
    print("White difficulty: ", calc_white_difficulty)
    print("Black difficulty: ", calc_black_difficulty)



def white_difficulty(board, move, white_material_delta, black_material_delta, white_bishop_pair_delta, black_bishop_pair_delta, white_king_pawn_distance_delta, 
               black_king_pawn_distance_delta, white_double_pawns_delta, black_double_pawns_delta, white_passed_pawns_delta, black_passed_pawns_delta, 
               white_total_control_delta, black_total_control_delta, white_weighted_bonus_delta, black_weighted_bonus_delta, is_sac_white, is_sac_black):
    # returns difficulty of the move that WHITE just played
    # white_material_delta --> positive if white gained material, negative if white lost material --> PROMOTION
    # black_material_delta --> positive if black gained material, negative if black lost material --> CAPTURE
    # white_bishop_pair_delta --> tuple (x, y) where x, y \in {True, False} where x is True if white has bishop pair before MOVE and y is True if white has bishop pair after MOVE
    # black_bishop_pair_delta --> tuple (x, y) where x, y \in {True, False} where x is True if black has bishop pair before MOVE and y is True if black has bishop pair after MOVE
    # white_king_pawn_distance_delta --> positive if white king moved closer to white pawns, negative if white king moved further from white pawns
    # black_king_pawn_distance_delta --> positive if black king moved closer to black pawns, negative if black king moved further from black pawns
    # white_double_pawns_delta --> positive if white gained double pawns, negative if white lost double pawns
    # black_double_pawns_delta --> positive if black gained double pawns, negative if black lost double pawns
    # white_passed_pawns_delta --> positive if white gained passed pawns, negative if white lost passed pawns
    # black_passed_pawns_delta --> positive if black gained passed pawns, negative if black lost passed pawns
    # white_total_control_delta --> positive if white gained control, negative if white lost control
    # black_total_control_delta --> positive if black gained control, negative if black lost control
    # white_weighted_bonus_delta --> positive if white gained bonus, negative if white lost bonus
    # black_weighted_bonus_delta --> positive if black gained bonus, negative if black lost bonus
    # is_sac_white --> True if white sacrificed a piece
    # is_sac_black --> True if black sacrificed a piece
    piece_value_endgame = {
        "p": 206,
        "n": 854,
        "b": 915,
        "r": 1380,
        "q": 2682,
        "k": 0
    } 
    # board.piece_at(square).symbol() == piece
    square = move.to_square

    board.push(move)
    # sac_diff = 20 if is_sac_white else 0
    sac_diff = 0
    if (is_sac_white):
        sac_diff = 20 * (piece_value_endgame[board.piece_at(square).symbol().lower()]/854)
    board.pop()
    weighted_diff = abs(white_weighted_bonus_delta) * 1.3 if white_weighted_bonus_delta > 0 else abs(white_weighted_bonus_delta) * 0.7
    control_diff = abs(white_total_control_delta) * 1.3 if white_total_control_delta > 0 else abs(white_total_control_delta) * 0.7
    passed_pawn_diff = abs(white_passed_pawns_delta) * 1.3 if white_passed_pawns_delta > 0 else abs(white_passed_pawns_delta) * 0.7
    double_pawn_diff = abs(white_double_pawns_delta) * 1.3 if white_double_pawns_delta > 0 else abs(white_double_pawns_delta) * 0.7
    king_pawn_diff = abs(white_king_pawn_distance_delta) * 1.3 if white_king_pawn_distance_delta > 0 else abs(white_king_pawn_distance_delta) * 0.7
    bishop_pair_diff = 20 if white_bishop_pair_delta[1] else 0
    material_diff = abs(white_material_delta) * 1.3 if white_material_delta > 0 else abs(white_material_delta) * 0.7
    capture_diff = abs(black_material_delta) * 1.3 if black_material_delta < 0 else abs(black_material_delta) * 0.7
    return sac_diff + weighted_diff + control_diff + passed_pawn_diff + double_pawn_diff + king_pawn_diff + bishop_pair_diff + material_diff + capture_diff

def black_difficulty(board, move, white_material_delta, black_material_delta, white_bishop_pair_delta, black_bishop_pair_delta, white_king_pawn_distance_delta,
                black_king_pawn_distance_delta, white_double_pawns_delta, black_double_pawns_delta, white_passed_pawns_delta, black_passed_pawns_delta,
                white_total_control_delta, black_total_control_delta, white_weighted_bonus_delta, black_weighted_bonus_delta, is_sac_white, is_sac_black):
     # returns difficulty of the move that BLACK just played
     # white_material_delta --> positive if white gained material, negative if white lost material --> CAPTURE
     # black_material_delta --> positive if black gained material, negative if black lost material --> PROMOTION
     # white_bishop_pair_delta --> tuple (x, y) where x, y \in {True, False} where x is True if white has bishop pair before MOVE and y is True if white has bishop pair after MOVE
     # black_bishop_pair_delta --> tuple (x, y) where x, y \in {True, False} where x is True if black has bishop pair before MOVE and y is True if black has bishop pair after MOVE
     # white_king_pawn_distance_delta --> positive if white king moved closer to white pawns, negative if white king moved further from white pawns
     # black_king_pawn_distance_delta --> positive if black king moved closer to black pawns, negative if black king moved further from black pawns
     # white_double_pawns_delta --> positive if white gained double pawns, negative if white lost double pawns
     # black_double_pawns_delta --> positive if black gained double pawns, negative if black lost double pawns
     # white_passed_pawns_delta --> positive if white gained passed pawns, negative if white lost passed pawns
     # black_passed_pawns_delta --> positive if black gained passed pawns, negative if black lost passed pawns
     # white_total_control_delta --> positive if white gained control, negative if white lost control
     # black_total_control_delta --> positive if black gained control, negative if black lost control
     # white_weighted_bonus_delta --> positive if white gained bonus, negative if white lost bonus
     # black_weighted_bonus_delta --> positive if black gained bonus, negative if black lost bonus
     # is_sac_white --> True if white sacrificed a piece
     # is_sac_black --> True if black sacrificed a piece

    piece_value_endgame = {
        "p": 206,
        "n": 854,
        "b": 915,
        "r": 1380,
        "q": 2682,
        "k": 0
    } 
    # board.piece_at(square).symbol() == piece
    square = move.to_square
    board.push(move)
    # sac_diff = 20 if is_sac_white else 0
    sac_diff = 0
    if (is_sac_black):
        sac_diff = 20 * (piece_value_endgame[board.piece_at(square).symbol().lower()]/854)
    board.pop()
    sac_diff = 20 if is_sac_black else 0
    weighted_diff = abs(black_weighted_bonus_delta) * 1.3 if black_weighted_bonus_delta > 0 else abs(black_weighted_bonus_delta) * 0.7
    control_diff = abs(black_total_control_delta) * 1.3 if black_total_control_delta > 0 else abs(black_total_control_delta) * 0.7
    passed_pawn_diff = abs(black_passed_pawns_delta) * 1.3 if black_passed_pawns_delta > 0 else abs(black_passed_pawns_delta) * 0.7
    double_pawn_diff = abs(black_double_pawns_delta) * 1.3 if black_double_pawns_delta > 0 else abs(black_double_pawns_delta) * 0.7

    king_pawn_diff = abs(black_king_pawn_distance_delta) * 1.3 if black_king_pawn_distance_delta > 0 else abs(black_king_pawn_distance_delta) * 0.7
    bishop_pair_diff = 20 if black_bishop_pair_delta[1] else 0
    material_diff = abs(black_material_delta) * 1.3 if black_material_delta > 0 else abs(black_material_delta) * 0.7
    capture_diff = abs(white_material_delta) * 1.3 if white_material_delta < 0 else abs(white_material_delta) * 0.7
    return sac_diff + weighted_diff + control_diff + passed_pawn_diff + double_pawn_diff + king_pawn_diff + bishop_pair_diff + material_diff + capture_diff












stockfish = Stockfish("/opt/homebrew/Cellar/stockfish/16/bin/stockfish", depth=23)
# read first game in ruy_lopezes.txt
# pgn = open("../../lichess_db_standard_rated_2017-02.pgn")
# pgn = open("single_game.pgn")
pgn = open("knight_sac.pgn")
# pgn = open("smol_game.pgn")
game = chess.pgn.read_game(pgn)

move_list = list(game.mainline_moves())
print(game.mainline_moves())
stockfish.set_position(move_list)
board = chess.Board()
for i in move_list:
    # print(type(i))
    board.push(i)
# get_all_heuristics(board)
print(board)
print(list(board.legal_moves))

# determine who is playing next

# get stockfish suggestions
time1 = time.time()
top_moves = stockfish.get_top_moves(20)
print(time.time() - time1)

print(top_moves)
get_all_delta_for_move(board, chess.Move.from_uci(str(top_moves[0]["Move"])))
print(attacks(board, chess.E3, chess.WHITE))


# print(material_count(stockfish.board()))

