import csv
from stockfish import Stockfish
import chess
from heuristics import (count_passed_pawn,
                        total_control,
                        weighted_bonus, count_black_double_pawns,
                        count_white_double_pawns, king_pawn_distance, black_has_bishop_pair, white_has_bishop_pair,
                        material_count)

parsed_puzzles = []

def parse_csv():
    with open('lichess_db_puzzle.csv', newline='') as file: 
        buffer = csv.reader(file, delimiter=",")
        for line in buffer:
            parsed_puzzles.append(line)
    print(parsed_puzzles)
    return parsed_puzzles
            
def generate_explanations():
    stockfish = Stockfish("/opt/homebrew/Cellar/stockfish/16/bin/stockfish", depth=23)
    for puzzle in parsed_puzzles:
        FEN = puzzle[1]
        moves = puzzle[2]
        move_list = moves.split()
        board = chess.Board(FEN)
        # push the first move (initial move prior to puzzle)
        board.push(move_list[0])
        starting_move_list = list(board.mainline_moves())
        turn = board.turn
        
        # get heuristics for starting position
        starting_passed_pawn = count_passed_pawn(board, turn)
        starting_total_control = total_control(board, turn)
        starting_weighted_bonus = weighted_bonus(board, turn)
        starting_black_double_pawns = count_black_double_pawns(board)
        starting_white_double_pawns = count_white_double_pawns(board)
        (starting_white_king_pawn_distance, starting_black_king_pawn_distance) = king_pawn_distance(board)
        starting_black_has_bishop_pair = black_has_bishop_pair(board)
        starting_white_has_bishop_pair = white_has_bishop_pair(board)
        (starting_white_material, starting_black_material) = material_count(board)

        # push the rest of the moves (this is all we're checking) 
        for i in range(len(move_list) - 1):
            board.push(move_list[i + 1])
        ending_move_list = list(board.mainline_moves())

        # check if leads to checkmate
        leads_to_mate = board.is_checkmate()

        # get heuristics for ending position
        ending_passed_pawn = count_passed_pawn(board, turn)
        ending_total_control = total_control(board, turn)
        ending_weighted_bonus = weighted_bonus(board, turn)
        ending_black_double_pawns = count_black_double_pawns(board)
        ending_white_double_pawns = count_white_double_pawns(board)
        (ending_white_king_pawn_distance, ending_black_king_pawn_distance) = king_pawn_distance(board)
        ending_black_has_bishop_pair = black_has_bishop_pair(board)
        ending_white_has_bishop_pair = white_has_bishop_pair(board)
        (ending_white_material, ending_black_material) = material_count(board)

        # determining most important heuristic 
        # 1. check if you checkmate 
        # 2. check for positive material change (if negative, we can add the phrase "even though you lose material" which is really cool)
        # 3. check for king safety 
        # 4. check if increased total control
        # 5. check if passed pawn
        # 6. check if opponent has doubled pawns 
        # 7. say it's because tempo 

        # for bad moves: 

        # 1. check if you missed a mate
        # 2. check if hanging mate
        # 3. check if hanging a piece (run low-depth stockfish and see if the piece is still on the board)
        # 4. check if king is in danger 
        # 5. check for losing control 
        # 6. say it's because tempo 

        # we'll generate two datapoints: correct move and incorrect move 
        # correct move 
        explanation = "This was a good move! After: " + moves
        # stockfish.set_position(starting_move_list)
        # starting_position = stockfish.get_evaluation()



