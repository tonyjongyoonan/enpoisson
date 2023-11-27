import chess.pgn
from stockfish import Stockfish
import chess
import torch
from heuristics import (is_passed_pawn_black, is_passed_pawn_white, count_passed_pawn, knight_attack, king_attack, pawn_attack, 
                        knight_control, pawn_control, king_control, bishop_xray_control, rook_xray_control, queen_control, total_control,
                        pawn_bonus, count_squares_knight_attacks, knight_bonus, weighted_bonus, pinned_direction, count_black_double_pawns,
                        count_white_double_pawns, king_pawn_distance, black_has_bishop_pair, white_has_bishop_pair, white_is_sac, black_is_sac,
                        material_count, get_white_material_delta, get_black_material_delta, white_delta_bishop_pair, black_delta_bishop_pair, 
                        white_delta_king_pawn_distance, black_delta_king_pawn_distance, white_delta_double_pawns, black_delta_double_pawns, 
                        black_delta_passed_pawns, white_delta_passed_pawns, white_delta_total_control, black_delta_total_control, white_delta_weighted_bonus,
                        black_delta_weighted_bonus)

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

    pgn_file = "lichess_db_standard_rated_2013-01.pgn"
    pgn = open(pgn_file)
    game = chess.pgn.read_game(pgn)

    #dataset of tuples ({}, label) where label is True if move was played and False otherwise
    dataset = []

    while (game is not None):

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
            features["white_material_count"] = material_count(board)[0]
            features["black_material_count"] = material_count(board)[1]
            features["white_bishop_pair"] = white_has_bishop_pair(board)
            features["black_bishop_pair"] = black_has_bishop_pair(board)
            features["white_king_pawn_dist"] = king_pawn_distance(board)[0]
            features["black_king_pawn_dist"] = king_pawn_distance(board)[1]
            features["white_doubled_pawns"] = count_white_double_pawns(board)
            features["black_doubled_pawns"] = count_black_double_pawns(board)
            features["white_passed_pawns"] = count_passed_pawn(board, chess.WHITE)
            features["black_passed_pawns"] = count_passed_pawn(board, chess.BLACK)
            features["white_total_control"] = total_control(board, chess.WHITE)
            features["black_total_control"] = total_control(board, chess.BLACK)
            features["white_weighted_bonus"] = weighted_bonus(board, chess.WHITE)
            features["black_weighted_bonus"] = weighted_bonus(board, chess.BLACK)
            # Pass in updated board
            move_ = chess.Move.from_uci(move["Move"])
            board.push(move_)
            features["new_board"] = board_to_bitboard_array(board)
            board.pop()
            # End
            label = int(features["move"].lower() == str(move_list[i + 1]).lower())
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
        
        game = chess.pgn.read_game(pgn)
        print(dataset)
    return dataset

def transform_data(raw):
    X = []
    Y = []
    for x,y in raw:
        X.append([int(x['elo'])] + x['board'] + x['new_board'])
        Y.append(y)
    return torch.tensor(X,dtype=torch.float32),torch.tensor(Y,dtype=torch.float32)