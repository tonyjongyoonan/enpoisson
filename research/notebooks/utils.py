import torch
from torch.utils.data import Dataset
import chess
import random

piece_to_index = {
    "p": 0,
    "r": 1,
    "b": 2,
    "n": 3,
    "q": 4,
    "k": 5,
}


def string_to_array(string):
    rows = string.split("/")
    ans = [[[0 for a in range(8)] for b in range(8)] for c in range(6)]
    for row in range(8):
        curr_row = rows[row]
        # print(curr_row)
        offset = 0
        for piece in range(len(curr_row)):
            curr_piece = curr_row[piece]
            sign = (
                1 if curr_piece.lower() == curr_piece else -1
            )  # check if the piece is capitalized
            curr_piece = (
                curr_piece.lower()
            )  # after storing whether or not capitalized, standardize it to lower case for easy processing
            if curr_piece not in piece_to_index.keys():
                offset += int(curr_piece) - 1
            else:
                current_board = ans[piece_to_index[curr_piece]]
                current_board[row][offset + piece] = 1 * sign
                ans[piece_to_index[curr_piece]] = current_board
    return ans


def string_to_array_two(string):
    rows = string.split("/")
    # Adjusted to 12 to account for separate layers for black and white pieces of each type
    ans = [[[0 for a in range(8)] for b in range(8)] for c in range(12)]
    # Mapping for pieces to their respective index in the 12 layer array
    # White pieces are in the first 6 layers and black pieces in the last 6 layers
    piece_to_index = {
        "p": 0,
        "r": 1,
        "n": 2,
        "b": 3,
        "q": 4,
        "k": 5,
        "P": 6,
        "R": 7,
        "N": 8,
        "B": 9,
        "Q": 10,
        "K": 11,
    }

    for row in range(8):
        curr_row = rows[row]
        offset = 0
        for piece in range(len(curr_row)):
            curr_piece = curr_row[piece]
            if curr_piece not in piece_to_index.keys():
                offset += int(curr_piece) - 1
            else:
                current_board = ans[piece_to_index[curr_piece]]
                current_board[row][offset + piece] = 1
                ans[piece_to_index[curr_piece]] = current_board
    # channels 13-20 are for the last 8 moves?
    # channels 21-22 is for castles

    return ans


""" This vocabulary is simply to turn the labels (predicted move) into integers which PyTorch Models can understand"""


class Vocabulary:
    def __init__(self):
        self.move_to_id = {"<UNK>": 0}
        self.id_to_move = {0: "<UNK>"}
        self.index = 1  # Start indexing from 1

    def add_move(self, move):
        if move not in self.move_to_id:
            self.move_to_id[move] = self.index
            self.id_to_move[self.index] = move
            self.index += 1

    def get_id(self, move):
        return self.move_to_id.get(move, self.move_to_id["<UNK>"])

    def get_move(self, id):
        return self.id_to_move.get(id, self.id_to_move[0])


class ChessDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features, label = self.X[idx], self.Y[idx]

        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            label, dtype=torch.long
        )


def df_to_data(df, sampling_rate=1.0, algebraic_notation=True):
    """
    Input: Dataframe of training data in which each row represents a full game played between players
    Output: List in which each item represents some game's history up until a particular move, List in the same order in which the associated label is the following move
    """
    board_states = []
    next_moves = []
    vocab = Vocabulary()
    chess_board = chess.Board()
    for game_board, game_moves in zip(df["board"], df["moves"]):
        moves = game_moves.split()
        boards = game_board.split("*")
        # Encode the moves into SAN notation and then into corresponding indices
        encoded_moves = []
        for move in moves:
            # Create a move object from the coordinate notation
            move_obj = chess.Move.from_uci(move)
            if move_obj not in chess_board.legal_moves:
                break
            else:
                if algebraic_notation:
                    algebraic_move = chess_board.san(move_obj)
                    chess_board.push(move_obj)
                    vocab.add_move(algebraic_move)
                    encoded_move = vocab.get_id(algebraic_move)
                    encoded_moves.append(encoded_move)
                else:
                    encoded_move = vocab.get_id(move)
                    encoded_moves.append(encoded_move)
        chess_board.reset()
        boards = boards[: len(encoded_moves)]
        # Now generate X,Y with sampling
        for i in range(len(encoded_moves) - 1):
            # TODO: Figure out how to deal with black orientation 'seeing' a different board
            if random.uniform(0, 1) <= sampling_rate and "w" in boards[i]:
                label = encoded_moves[i + 1]
                board_states.append(string_to_array(boards[i].split(" ")[0]))
                next_moves.append(label)

    return board_states, next_moves, vocab


# Function to calculate top-3 accuracy
def top_3_accuracy(y_true, y_pred):
    top3 = torch.topk(y_pred, 3, dim=1).indices
    correct = top3.eq(y_true.view(-1, 1).expand_as(top3))
    return correct.any(dim=1).float().mean().item()
