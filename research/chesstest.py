from stockfish import Stockfish
# import chess

stockfish = Stockfish("/opt/homebrew/Cellar/stockfish/16/bin/stockfish", depth=23)
stockfish.set_depth(23)
# stockfish.set_skill_level(20)
setup = ["e2e4", "e7e5", "g1f3"]
stockfish.set_position(setup)
# stockfish.evalType = "NNUE"
stockfish.BenchmarkParameters.evalType = "NNUE"
print(stockfish.BenchmarkParameters.evalType)



print(stockfish.BenchmarkParameters())


for i in stockfish.get_top_moves(3):
    print("------------------")
    print("Move: ", i)
    setup_copy = setup.copy()
    setup_copy.append(i['Move'])
    stockfish.set_position(setup_copy)
    print(stockfish.get_board_visual())
    print("evaluation: ", stockfish.get_evaluation()['value']/100)   


