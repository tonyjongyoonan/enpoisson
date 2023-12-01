import chess.pgn
from stockfish import Stockfish
import chess
import torch
import random
import numpy as np
from numpy.random import choice
"""from heuristics import (is_passed_pawn_black, is_passed_pawn_white, count_passed_pawn, knight_attack, king_attack, pawn_attack, 
                        knight_control, pawn_control, king_control, bishop_xray_control, rook_xray_control, queen_control, total_control,
                        pawn_bonus, count_squares_knight_attacks, knight_bonus, weighted_bonus, pinned_direction, count_black_double_pawns,
                        count_white_double_pawns, king_pawn_distance, black_has_bishop_pair, white_has_bishop_pair, white_is_sac, black_is_sac,
                        material_count, get_white_material_delta, get_black_material_delta, white_delta_bishop_pair, black_delta_bishop_pair, 
                        white_delta_king_pawn_distance, black_delta_king_pawn_distance, white_delta_double_pawns, black_delta_double_pawns, 
                        black_delta_passed_pawns, white_delta_passed_pawns, white_delta_total_control, black_delta_total_control, white_delta_weighted_bonus,
                        black_delta_weighted_bonus)
"""
def board_to_bitboard_array(board):
    bitboard_array = []
    for piece_type in chess.PIECE_TYPES:
        for color in chess.COLORS:
            # Get the bitboard for each piece type and color
            bitboard = board.pieces_mask(piece_type, color)
            # Convert the bitboard to a list of bits and extend the bitboard_array
            bitboard_array.extend([int(bool(bitboard & (1 << square))) for square in range(64)])
    return bitboard_array

def get_training_data_raw(num_games, validation=False):
    stockfish = Stockfish("/opt/homebrew/Cellar/stockfish/16/bin/stockfish", depth=23)
    stockfish.set_depth(5)

    pgn_file = "lichess_db_standard_rated_2013-01.pgn"
    pgn = open(pgn_file)
    game = chess.pgn.read_game(pgn)

    #dataset of tuples ({}, label) where label is True if move was played and False otherwise
    dataset = []
    count = 0
    if validation: 
        for i in range(100):
            game = chess.pgn.read_game(pgn)
    while (game is not None) and count < num_games:
        move_list = list(game.mainline_moves())
        stockfish.set_position(move_list)

        #start one position in since openings are whatever
        curr_move = "Black"
        tmp_moves = [move_list[0]]

        # Duplicate board representation onto board object
        board = chess.Board()
        board.push(tmp_moves[-1])

        for i in range(len(move_list) - 1):
            stockfish.set_position(tmp_moves)
            # feature set 
            legal_moves = board.legal_moves
            all_moves = stockfish.get_top_moves(len(list(legal_moves)))
            eval = []
            played_move = None
            for move in all_moves:
                if move["Move"].lower() == str(move_list[i + 1]).lower():
                    played_move = move
                if move["Centipawn"] is None:
                    eval.append(0)
                else:
                    eval.append(move["Centipawn"])
            softmax = torch.nn.Softmax(dim=-1)
            input = torch.tensor(eval)
            input = input.float()
            input = input / 1000
            probabilities = softmax(input)
            probabilities = np.array(list(probabilities))
            probabilities /= probabilities.sum()
            moves_to_use = choice(all_moves, min([4, np.count_nonzero(probabilities)]), replace=False, p=probabilities)
            moves_to_use = list(moves_to_use)
            counter = False
            for move in moves_to_use:
                if move["Move"].lower() == str(move_list[i + 1]).lower():
                    counter = True
            if not counter:
                moves_to_use.append(played_move)
            # moves_to_use.append(move_list[i + 1])
            curr_board = board_to_bitboard_array(board)
            """
            white_material_count = material_count(board)[0]
            black_material_count = material_count(board)[1]
            white_bishop_pair = white_has_bishop_pair(board)
            black_bishop_pair = black_has_bishop_pair(board)
            white_king_pawn_dist = king_pawn_distance(board)[0]
            black_king_pawn_dist = king_pawn_distance(board)[1]
            white_doubled_pawns = count_white_double_pawns(board)
            black_doubled_pawns = count_black_double_pawns(board)
            white_passed_pawns = count_passed_pawn(board, chess.WHITE)
            black_passed_pawns = count_passed_pawn(board, chess.BLACK)
            white_total_control = total_control(board, chess.WHITE)
            black_total_control = total_control(board, chess.BLACK)
            white_weighted_bonus = weighted_bonus(board, chess.WHITE)
            black_weighted_bonus = weighted_bonus(board, chess.BLACK)
            """
            for move in moves_to_use:
                features = {}
                features["elo"] = game.headers[curr_move + "Elo"]
                features["board"] = curr_board
                features["move"] = str(move["Move"])
                """
                features["centipawn"] = move["Centipawn"]
                features["white_material_count"] = white_material_count
                features["black_material_count"] = black_material_count
                features["white_bishop_pair"] = white_bishop_pair
                features["black_bishop_pair"] = black_bishop_pair
                features["white_king_pawn_dist"] = white_king_pawn_dist
                features["black_king_pawn_dist"] = black_king_pawn_dist
                features["white_doubled_pawns"] = white_doubled_pawns
                features["black_doubled_pawns"] = black_doubled_pawns
                features["white_passed_pawns"] = white_passed_pawns
                features["black_passed_pawns"] = black_passed_pawns
                features["white_total_control"] = white_total_control
                features["black_total_control"] = black_total_control
                features["white_weighted_bonus"] = white_weighted_bonus
                features["black_weighted_bonus"] = black_weighted_bonus
                """
                # Pass in updated board
                move_ = chess.Move.from_uci(move["Move"])
                board.push(move_)
                features["new_board"] = board_to_bitboard_array(board)
                board.pop()
                # Undo update on board
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
        count += 1
        print(count)
    return dataset

def transform_data(raw):
    X = []
    Y = []
    for x,y in raw:
        if '?' not in x['elo']:
            X.append([int(x['elo'])] + x['board'] + x['new_board'])
            Y.append(y)
    return torch.tensor(X,dtype=torch.float32),torch.tensor(Y,dtype=torch.long)
