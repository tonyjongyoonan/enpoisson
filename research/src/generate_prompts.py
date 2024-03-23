import json
import chess
from heuristics import (count_passed_pawn,
                        total_control,
                        weighted_bonus, count_black_double_pawns,
                        count_white_double_pawns, king_pawn_distance, black_has_bishop_pair, white_has_bishop_pair,
                        material_count, check_hanging)


def convert_board_to_string(board):
    #takes in 8 by 8 board state with 'p', '.', etc, and converts to string 'pppppppp/8/8/...' format
    string_rep = ""
    for i in range(8):
        period_counter = 0
        for j in range(8):
            curr_val = board[i][j]
            if (curr_val == '.'):
                period_counter += 1
            else:
                if (period_counter > 0):
                    string_rep += str(period_counter)
                    period_counter = 0
                string_rep += curr_val
        if (period_counter > 0):
            string_rep += str(period_counter)
        string_rep += "/"
    return string_rep[:-1]

# def print_board(board):
#     string_rep = "[["
#     for i in range(8):
#         for j in range(8):
#             string_rep += "'"
#             string_rep += board[i][j]


def generate_prompts():
    file = "../cleansed_llm_data.txt"
    all_moves = []

    with open(file) as f:
        moves = f.readlines()
    
    for move in moves:
        move = move.replace("\'start_state\'", "\"start_state\"")
        move = move.replace("\'end_state\'", "\"end_state\"")
        move = move.replace("\'moves\': \'", "\"moves\": \"")
        move = move.replace("\', \'commentary\': \'", "\", \"commentary\": \"")
        move = move.replace("\', \'commentary\'", "\", \"commentary\"")
        move = move.replace("\'}", "\"}")
        move = move.replace("\'r\'", "\"r\"")
        move = move.replace("\'n\'", "\"n\"")
        move = move.replace("\'b\'", "\"b\"")
        move = move.replace("\'q\'", "\"q\"")
        move = move.replace("\'k\'", "\"k\"")
        move = move.replace("\'p\'", "\"p\"")
        move = move.replace("\'R\'", "\"R\"")
        move = move.replace("\'N\'", "\"N\"")
        move = move.replace("\'B\'", "\"B\"")
        move = move.replace("\'Q\'", "\"Q\"")
        move = move.replace("\'K\'", "\"K\"")
        move = move.replace("\'P\'", "\"P\"")
        move = move.replace("\'.\'", "\".\"")






        try:
            new_move = json.loads(move)
        except json.decoder.JSONDecodeError:
            print("error")
        all_moves.append(new_move)
    
    compiled_dataset = []
    for move in all_moves:
        #for use in heuristic functions 
        start_board = convert_board_to_string(move.get("start_state"))
        end_board = convert_board_to_string(move.get("end_state"))
        start = chess.Board(start_board)
        end = chess.Board(end_board)
        # basically just check if the moves string looks like "1. e4 e5" in which case first move was white or "1... e5" in which case first move was black
        # kind of a dumb way to do it but it should work as long as none of the games have triple digit moves LOL
        turn_to_move = "black" if move.get("moves")[3] == '.' else "white"
        # get heuristics here 
        start_white_passed_pawns = count_passed_pawn(start, chess.WHITE)
        start_black_passed_pawns = count_passed_pawn(start, chess.BLACK)
        end_white_passed_pawns = count_passed_pawn(end, chess.WHITE)
        end_black_passed_pawns = count_passed_pawn(end, chess.BLACK)

        start_white_total_control = total_control(start, chess.WHITE)
        start_black_total_control = total_control(start, chess.BLACK)
        end_white_total_control = total_control(end, chess.WHITE)
        end_black_total_control = total_control(end, chess.BLACK)

        start_white_double_pawns = count_white_double_pawns(start) 
        start_black_double_pawns = count_black_double_pawns(start)
        end_white_double_pawns = count_white_double_pawns(end)
        end_black_double_pawns = count_black_double_pawns(end)

        (start_white_king_pawn_distance, start_black_king_pawn_distance) = king_pawn_distance(start)
        (end_white_king_pawn_distance, end_black_king_pawn_distance) = king_pawn_distance(end)

        start_white_has_bishop_pair = white_has_bishop_pair(start)
        start_black_has_bishop_pair = black_has_bishop_pair(start)
        end_white_has_bishop_pair = white_has_bishop_pair(end)
        end_black_has_bishop_pair = black_has_bishop_pair(end)

        (start_white_material_count, start_black_material_count) = material_count(start)
        (end_white_material_count, end_black_material_count) = material_count(end)




        prompt = "You're a chess grandmaster analyzing a series of moves from " + turn_to_move + "'s perspective. The starting position of the board looks like this: "
        prompt += str(move.get("start_state")) + ". It's " + turn_to_move + "'s turn to play. The following moves are played: " + move.get("moves")
        prompt += ". After this series of moves, the new board position looks like this: " + str(move.get("end_state"))
        prompt += ". Analyze this series of moves from " + turn_to_move + "'s perspective. You may use the following heuristics to help your answer: "
        prompt += "White material count at start state: " + str(start_white_material_count) 
        prompt += ". Black material count at start state: " + str(start_black_material_count) 
        prompt += ". White material count at end state: " + str(end_white_material_count) 
        prompt += ". Black material count at end state: " + str(end_black_material_count) 
        prompt += ". Total number of White passed pawns at start state: " + str(start_white_passed_pawns)
        prompt += ". Total number of Black passed pawns at start state: " + str(start_black_passed_pawns)
        prompt += ". Total number of White passed pawns at end state: " + str(end_white_passed_pawns)
        prompt += ". Total number of Black passed pawns at end state: " + str(end_black_passed_pawns)
        prompt += ". Total number of White doubled pawns at start state: " + str(start_white_double_pawns)
        prompt += ". Total number of Black doubled pawns at start state: " + str(start_black_double_pawns)
        prompt += ". Total number of White doubled pawns at end state: " + str(end_white_double_pawns)
        prompt += ". Total number of Black doubled pawns at end state: " + str(end_black_double_pawns)
        prompt += ". Total control White has at start state: " + str(start_white_total_control)
        prompt += ". Total control Black has at start state: " + str(start_black_total_control)
        prompt += ". Total control White has at end state: " + str(end_white_total_control)
        prompt += ". Total control Black has at end state: " + str(end_black_total_control)
        prompt += ". Does White have a bishop pair at the start state? " + str(start_white_has_bishop_pair)
        prompt += ". Does Black have a bishop pair at the start state? " + str(start_black_has_bishop_pair)
        prompt += ". Does White have a bishop pair at the end state? " + str(end_white_has_bishop_pair)
        prompt += ". Does Black have a bishop pair at the end state? " + str(end_black_has_bishop_pair)
        prompt += ". Distance between White king and the closest White pawn at start state: " + str(start_white_king_pawn_distance)
        prompt += ". Distance between Black king and the closest Black pawn at start state: " + str(start_black_king_pawn_distance)
        prompt += ". Distance between White king and the closest Black pawn at end state: " + str(end_white_king_pawn_distance)
        prompt += ". Distance between Black king and the closest Black pawn at end state: " + str(end_black_king_pawn_distance)

        new_obj = {"input_text": prompt, "output_text": move.get("commentary")}
        compiled_dataset.append(new_obj)
        print(new_obj)


    with open("data_1.json", "w") as f:
        for item in compiled_dataset:
            f.write(json.dumps(item) + "\n")



# final function-- takes in start and end board as an FEN string, move as a string to analyze 
# turn is chess.WHITE or chess.BLACK
def generate_prompt(start_board, move, end_board, turn): 
    start = chess.Board(start_board)
    end = chess.Board(end_board)

    turn_to_move = "white" if turn == chess.WHITE else "black"
    
    # get heuristics here 
    start_white_passed_pawns = count_passed_pawn(start, chess.WHITE)
    start_black_passed_pawns = count_passed_pawn(start, chess.BLACK)
    end_white_passed_pawns = count_passed_pawn(end, chess.WHITE)
    end_black_passed_pawns = count_passed_pawn(end, chess.BLACK)

    start_white_total_control = total_control(start, chess.WHITE)
    start_black_total_control = total_control(start, chess.BLACK)
    end_white_total_control = total_control(end, chess.WHITE)
    end_black_total_control = total_control(end, chess.BLACK)

    start_white_double_pawns = count_white_double_pawns(start) 
    start_black_double_pawns = count_black_double_pawns(start)
    end_white_double_pawns = count_white_double_pawns(end)
    end_black_double_pawns = count_black_double_pawns(end)

    (start_white_king_pawn_distance, start_black_king_pawn_distance) = king_pawn_distance(start)
    (end_white_king_pawn_distance, end_black_king_pawn_distance) = king_pawn_distance(end)

    start_white_has_bishop_pair = white_has_bishop_pair(start)
    start_black_has_bishop_pair = black_has_bishop_pair(start)
    end_white_has_bishop_pair = white_has_bishop_pair(end)
    end_black_has_bishop_pair = black_has_bishop_pair(end)

    (start_white_material_count, start_black_material_count) = material_count(start)
    (end_white_material_count, end_black_material_count) = material_count(end)

    start_white_hanging = check_hanging(start, turn)
    end_white_hanging = check_hanging(end, turn)
    start_black_hanging = check_hanging(start, not turn)
    end_black_hanging = check_hanging(end, not turn)


    prompt = "You're a chess grandmaster analyzing a move from " + turn_to_move + "'s perspective. The starting position of the board looks like this: "
    prompt += start_board + ". It's " + turn_to_move + "'s turn to play. The following move is played: " + move
    prompt += ". After this move, the new board position looks like this: " + end_board
    prompt += ". Analyze this move from " + turn_to_move + "'s perspective. You may use the following heuristics to help your answer: "
    prompt += "White material count at start state: " + str(start_white_material_count) 
    prompt += ". Black material count at start state: " + str(start_black_material_count) 
    prompt += ". White material count at end state: " + str(end_white_material_count) 
    prompt += ". Black material count at end state: " + str(end_black_material_count) 
    prompt += ". Total number of White passed pawns at start state: " + str(start_white_passed_pawns)
    prompt += ". Total number of Black passed pawns at start state: " + str(start_black_passed_pawns)
    prompt += ". Total number of White passed pawns at end state: " + str(end_white_passed_pawns)
    prompt += ". Total number of Black passed pawns at end state: " + str(end_black_passed_pawns)
    prompt += ". Total number of White doubled pawns at start state: " + str(start_white_double_pawns)
    prompt += ". Total number of Black doubled pawns at start state: " + str(start_black_double_pawns)
    prompt += ". Total number of White doubled pawns at end state: " + str(end_white_double_pawns)
    prompt += ". Total number of Black doubled pawns at end state: " + str(end_black_double_pawns)
    prompt += ". Total control White has at start state: " + str(start_white_total_control)
    prompt += ". Total control Black has at start state: " + str(start_black_total_control)
    prompt += ". Total control White has at end state: " + str(end_white_total_control)
    prompt += ". Total control Black has at end state: " + str(end_black_total_control)
    prompt += ". Does White have a bishop pair at the start state? " + str(start_white_has_bishop_pair)
    prompt += ". Does Black have a bishop pair at the start state? " + str(start_black_has_bishop_pair)
    prompt += ". Does White have a bishop pair at the end state? " + str(end_white_has_bishop_pair)
    prompt += ". Does Black have a bishop pair at the end state? " + str(end_black_has_bishop_pair)
    prompt += ". Distance between White king and the closest White pawn at start state: " + str(start_white_king_pawn_distance)
    prompt += ". Distance between Black king and the closest Black pawn at start state: " + str(start_black_king_pawn_distance)
    prompt += ". Distance between White king and the closest Black pawn at end state: " + str(end_white_king_pawn_distance)
    prompt += ". Distance between Black king and the closest Black pawn at end state: " + str(end_black_king_pawn_distance)
    prompt += ". Number of White pieces hanging at start state: " + str(start_white_hanging) 
    prompt += ". Number of White pieces hanging at end state: " + str(end_white_hanging) 
    prompt += ". Number of Black pieces hanging at start state: " + str(start_black_hanging) 
    prompt += ". Number of Black pieces hanging at end state: " + str(end_black_hanging) 

    return prompt
