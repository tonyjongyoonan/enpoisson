import json
# from heuristics import (count_passed_pawn,
#                         total_control,
#                         weighted_bonus, count_black_double_pawns,
#                         count_white_double_pawns, king_pawn_distance, black_has_bishop_pair, white_has_bishop_pair, white_is_sac, black_is_sac,
#                         material_count, get_white_material_delta, get_black_material_delta, white_delta_bishop_pair, black_delta_bishop_pair, 
#                         white_delta_king_pawn_distance, black_delta_king_pawn_distance, white_delta_double_pawns, black_delta_double_pawns, 
#                         black_delta_passed_pawns, white_delta_passed_pawns, white_delta_total_control, black_delta_total_control, white_delta_weighted_bonus,
#                         black_delta_weighted_bonus)


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
        # basically just check if the moves string looks like "1. e4 e5" in which case first move was white or "1... e5" in which case first move was black
        # kind of a dumb way to do it but it should work as long as none of the games have triple digit moves LOL
        turn_to_move = "black" if move.get("moves")[3] == '.' else "white"
        # get heuristics here 
        prompt = "You're a chess grandmaster analyzing a series of moves from " + turn_to_move + "\'s perspective. The starting position of the board looks like this: \n"
        prompt += str(move.get("start_state")) + "\nIt's " + turn_to_move + "\'s turn to play. The following moves are played: " + move.get("moves")
        prompt += "\nAfter this series of moves, the new board position looks like this: \n" + str(move.get("end_state"))
        prompt += "\nAnalyze this series of moves from " + turn_to_move + "\'s perspective. You may use the following heuristics to help your answer: "
        #prompt += heuristics lol 


    print(prompt)