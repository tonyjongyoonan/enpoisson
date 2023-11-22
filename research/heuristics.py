import chess.pgn
from stockfish import Stockfish
import chess

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


def pawn_attack(board, square, color):
    # returns True if SQUARE is attacked by opposing color pawn
    # NOTE: this function does not check if the square is occupied by a piece

    if (color == chess.WHITE):
        if (get_column(square) <= 6 and get_row(square) <= 6 and board.piece_at(square+9) != None):
            if (board.piece_at(square+9).symbol() == "p"):
                return True
        if (get_column(square) >= 1 and get_row(square) <= 6 and board.piece_at(square+7) != None):
            if (board.piece_at(square+7).symbol() == "p"):
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
        s = tmp + "\n" + s
    print(s)


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
            # TODO: rewrite to handle OOB wraparound nonsense. 
            if (get_row(square) + d * ix <= 7) and (get_row(square) + d * ix >= 0) and (get_column(square) + d * iy <= 7) and (get_column(square) + d * iy >= 0):
                if board.piece_at(square + d * ix + d * iy * 8) is not None:
                    if board.piece_at(square + d * ix + d * iy * 8).symbol() == "K":
                        king = True
                    break
        if king:
            for d in range(1, 8):
                if (get_row(square) - d * ix <= 7) and (get_row(square) - d * ix >= 0) and (get_column(square) - d * iy <= 7) and (get_column(square) - d * iy >= 0):
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
# print(material_count(stockfish.board()))
