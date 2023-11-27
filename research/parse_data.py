import chess.pgn
from stockfish import Stockfish
import chess
import torch

def board_to_bitboard_array(board):
    bitboard_array = []
    for piece_type in chess.PIECE_TYPES:
        for color in chess.COLORS:
            # Get the bitboard for each piece type and color
            bitboard = board.pieces_mask(piece_type, color)
            # Convert the bitboard to a list of bits and extend the bitboard_array
            bitboard_array.extend([int(bool(bitboard & (1 << square))) for square in range(64)])
    return bitboard_array

def get_training_data_raw():
    stockfish = Stockfish("/opt/homebrew/Cellar/stockfish/16/bin/stockfish", depth=23)
    stockfish.set_depth(8)

    pgn_file = "../lichess_db_standard_rated_2013-01.pgn"
    pgn = open(pgn_file)
    game = chess.pgn.read_game(pgn)

    move_list = list(game.mainline_moves())
    stockfish.set_position(move_list)

    #start one position in since openings are whatever
    curr_move = "Black"
    tmp_moves = [move_list[0]]

    # Duplicate board representation onto board object
    board = chess.Board()
    board.push(tmp_moves[-1])

    #dataset of tuples ({}, label) where label is True if move was played and False otherwise
    dataset = []

    for i in range(len(move_list) - 1):
        stockfish.set_position(tmp_moves)
        # feature set 
        top_moves = stockfish.get_top_moves(10)
        curr_board = board_to_bitboard_array(board)
        for move in top_moves:
            features = {}
            features["elo"] = game.headers[curr_move + "Elo"]
            features["board"] = curr_board
            features["move"] = str(move["Move"])
            # Pass in updated board
            move_ = chess.Move.from_uci(move["Move"])
            board.push(move_)
            features["new_board"] = board_to_bitboard_array(board)
            board.pop()
            # End
            label = features["move"].lower() == str(move_list[i + 1]).lower()
            dataset.append((features, label))
            if label:
                break

        # Update Board
        tmp_moves.append(move_list[i + 1])
        board.push(tmp_moves[-1])

        if curr_move == "White":
            curr_move = "Black"
        else:
            curr_move = "White"
    return dataset

def transform_data(raw):
    X = []
    Y = []
    for x,y in raw:
        X.append([x['elo']] + x['board'] + x['new_board'])
        Y.append(y)
    return X,Y

def batch_generator(X, Y, batch_size):
    for i in range((X.shape[0] - batch_size) // batch_size):
        yield torch.tensor(X[i * batch_size: i * batch_size + batch_size]),torch.tensor(Y[i * batch_size: i * batch_size + batch_size])
