import torch
import matplotlib.pyplot as plt
import chess
import torch.nn as nn
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import chess
import random
import numpy as np
import dask.dataframe as dd 
from torch.utils.data import Dataset
from bitarray import bitarray

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Objects """
        
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

class MultimodalDataset(Dataset):
    def __init__(self, sequences, boards, lengths, labels):
        self.sequences = sequences
        self.boards = boards
        self.lengths = lengths
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.boards[idx], dtype=torch.float32),
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.lengths[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
class MultimodalDatasetWithFEN(Dataset):
    def __init__(self, sequences, boards, lengths, fens, labels):
        self.sequences = sequences
        self.boards = boards
        self.lengths = lengths
        self.labels = labels
        self.fens = fens

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.boards[idx], dtype=torch.float32),
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.lengths[idx], dtype=torch.long),
            self.fens[idx],
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
    
class MultimodalTwoDataset(Dataset):
    def __init__(self, sequences, boards, lengths, labels):
        self.sequences = sequences
        self.boards = boards
        self.lengths = lengths
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.boards[idx], dtype=torch.float32),
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.lengths[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

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

class VocabularyWithCLS:
    def __init__(self):
        self.move_to_id = {"<UNK>": 0, "CLS": 1}
        self.id_to_move = {0: "<UNK>", 1: "CLS"}
        self.index = 2  # Start indexing from 2

    def add_move(self, move):
        if move not in self.move_to_id:
            self.move_to_id[move] = self.index
            self.id_to_move[self.index] = move
            self.index += 1

    def get_id(self, move):
        return self.move_to_id.get(move, self.move_to_id["<UNK>"])

    def get_move(self, id):
        return self.id_to_move.get(id, self.id_to_move[0])

class VocabularyForTransformer:
    def __init__(self):
        self.word_to_id = {"<PAD>": 0, "<START>": 1, "<BOARD>": 2, "<MOVE>": 3, "<SEP>": 4, "<CLS>": 5}
        self.id_to_word = {0: "<PAD>", 1: "<START>", 2: "<BOARD>", 3: "<MOVE>", 4: "<SEP>", 5: "<CLS>"}
        self.num_words = 5

    def add_move(self, word):
        if word not in self.word_to_id:
            self.word_to_id[word] = self.num_words
            self.id_to_word[self.num_words] = word
            self.num_words += 1

    def get_id(self, word):
        return self.word_to_id.get(word, None)

    def get_word(self, word_id):
        return self.id_to_word.get(word_id, None)
    
""" Data Processing """

def process_raw_csv(filepath):
    # Import CSV File (from Maia: http://csslab.cs.toronto.edu/datasets/#monthly_chess_csv)
    # The CSV has 151,072,060 rows
    data_types ={'clock': 'float32',
        'cp': 'object',
        'opp_clock': 'float32',
        'opp_clock_percent': 'float32'}
    df = dd.read_csv(filepath, blocksize='64e6', dtype= data_types, low_memory=False)

    # Filter out quick games (Bullet and HyperBullet) and take out moves that happened in the last XX seconds (this won't affect how many games we import but the # of moves we look at)
    condition_time_control = ~df['time_control'].isin(['Bullet', 'HyperBullet'])
    condition_clock = df['clock'] > 45
    # condition_plays = df['num_ply'] < 80
    filtered_df = df[condition_time_control & condition_clock]

    # Select Relevant Columns
    selected_columns = ['game_id','white_elo','black_elo','move','white_active','board']
    filtered_df = filtered_df[selected_columns]

    # Filter only games of Elo 1100-1199
    filtered_df = filtered_df[(filtered_df['white_elo'].between(1100, 1199)) & (filtered_df['black_elo'].between(1100, 1199))]

    # Group Same Games Together 
    def aggregate_moves(group):
        moves = ' '.join(group['move'])  # Concatenate moves into a single string
        white_elo = group['white_elo'].iloc[0]  # Get the first white_elo
        black_elo = group['black_elo'].iloc[0]  # Get the first black_elo
        white_active = group['white_active'].iloc[0]  # Get the first num_ply
        board = '*'.join(group['board'])  # Get the first num_ply
        return pd.Series({'moves': moves, 'white_elo': white_elo, 'black_elo': black_elo, 'white_active': white_active, 'board': board})

    grouped_df = filtered_df.groupby('game_id',sort=True).apply(aggregate_moves, meta={'moves': 'str', 'white_elo': 'int', 'black_elo': 'int', 'white_active': 'str', 'board': 'str'}).compute()

    # This gives us 99,300 Games when we don't filter games with more than 80 half-moves
    return grouped_df

def construct_vocab(df, algebraic_notation = True):
    vocab = Vocabulary()
    for game_board, game_moves in zip(df["board"], df["moves"]):
        moves = game_moves.split()
        if algebraic_notation:
            pass
    return vocab
        # encoded_moves = []
        # for move in moves:
        #     # Create a move object from the coordinate notation
        #     move_obj = chess.Move.from_uci(move)
        #     if move_obj not in chess_board.legal_moves:
        #         break
        #     else:
        #         if algebraic_notation:
        #             algebraic_move = chess_board.san(move_obj)
        #             chess_board.push(move_obj)
        #             vocab.add_move(algebraic_move)
        #             encoded_move = vocab.get_id(algebraic_move)
        #             encoded_moves.append(encoded_move)
        #         else:
        #             vocab.add_move(move)
        #             encoded_move = vocab.get_id(move)
        #             encoded_moves.append(encoded_move)
        # chess_board.reset()
        # del moves
        # boards = boards[: len(encoded_moves)]
        # # Now generate X,Y with sampling
        # for i in range(len(encoded_moves) - 1):
        #     # TODO: Figure out how to deal with black orientation 'seeing' a different board
        #     if random.uniform(0, 1) <= sampling_rate:
        #         label = encoded_moves[i + 1]
        #         fens.append(boards[i])
        #         board_states.append(fen_to_array_three(boards[i]))
        #         next_moves.append(label)
def fen_to_array(string):
    piece_to_index = {
        "p": 0,
        "r": 1,
        "b": 2,
        "n": 3,
        "q": 4,
        "k": 5,
    }
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


def fen_to_array_two(string):
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

def fen_to_array_three(string):
    board = string.split(" ")[0]
    rows = board.split("/")
    parts = string.split(' ')
    # Adjusted to 17 to account for separate layers for black and white pieces of each type
    ans = [[[False for _ in range(8)] for _ in range(8)] for _ in range(17)]
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
                current_board[row][offset + piece] = True
                ans[piece_to_index[curr_piece]] = current_board
    # Channel for the player to move (True if white's move, False if black's)
    ans[12] = [[True if parts[1] == 'w' else False for _ in range(8)] for _ in range(8)]

    # Castling rights
    castling = parts[2]
    ans[13] = [[True if 'K' in castling else False for _ in range(8)] for _ in range(8)]
    ans[14] = [[True if 'Q' in castling else False for _ in range(8)] for _ in range(8)]
    ans[15] = [[True if 'k' in castling else False for _ in range(8)] for _ in range(8)]
    ans[16] = [[True if 'q' in castling else False for _ in range(8)] for _ in range(8)]

    return ans
    
def df_to_data_board_only(df, sampling_rate=1.0, algebraic_notation=True, vocab_old = None):
    """
    Input: Dataframe of training data in which each row represents a full game played between players
    Output: List in which each item represents some game's history up until a particular move, List in the same order in which the associated label is the following move
    """
    board_states = []
    next_moves = []
    fens = []
    vocab = vocab_old
    if vocab is None:
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
            if algebraic_notation and (move_obj not in chess_board.legal_moves):
                break
            else:
                if algebraic_notation:
                    algebraic_move = chess_board.san(move_obj)
                    chess_board.push(move_obj)
                    vocab.add_move(algebraic_move)
                    encoded_move = vocab.get_id(algebraic_move)
                    encoded_moves.append(encoded_move)
                else:
                    vocab.add_move(move)
                    encoded_move = vocab.get_id(move)
                    encoded_moves.append(encoded_move)
        if algebraic_notation:
            chess_board.reset()
        del moves
        boards = boards[: len(encoded_moves)]
        # Now generate X,Y with sampling
        for i in range(len(encoded_moves) - 1):
            # TODO: Figure out how to deal with black orientation 'seeing' a different board
            if random.uniform(0, 1) <= sampling_rate:
                label = encoded_moves[i + 1]
                fens.append(boards[i])
                board_states.append(fen_to_array_three(boards[i]))
                next_moves.append(label)
    del chess_board
    return np.asarray(board_states,dtype=np.bool_), fens, np.asarray(next_moves), vocab

def df_to_data_simple(
    df,
    sampling_rate=1,
    fixed_window=True,
    fixed_window_size=16,
    algebraic_notation=True,
):
    """
    Input: Dataframe of training data in which each row represents a full game played between players
    Output: List in which each item represents some game's history up until a particular move, List in the same order in which the associated label is the following move
    """
    board_states, subsequences, next_moves = [], [], []
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
            # There are some broken moves in the data -> stop reading if so
            if algebraic_notation and move_obj not in chess_board.legal_moves:
                break
            else:
                if algebraic_notation:
                    algebraic_move = chess_board.san(move_obj)
                    chess_board.push(move_obj)
                    vocab.add_move(algebraic_move)
                    encoded_move = vocab.get_id(algebraic_move)
                    encoded_moves.append(encoded_move)
                else:
                    vocab.add_move(move)
                    encoded_move = vocab.get_id(move)
                    encoded_moves.append(encoded_move)
        chess_board.reset()
        boards = boards[: len(encoded_moves)]
        # Now generate X,Y with sampling
        for i in range(len(encoded_moves) - 1):
            # TODO: Figure out how to deal with black orientation 'seeing' a different board
            if random.uniform(0, 1) <= sampling_rate:
                # Board
                board_states.append(fen_to_array_two(boards[i].split(" ")[0]))
                # Sequence of Moves
                subseq = encoded_moves[0 : i + 1]
                if fixed_window and len(subseq) > fixed_window_size:
                    subseq = subseq[-fixed_window_size:]
                subsequences.append(subseq)
                # Label
                label = encoded_moves[i + 1]
                next_moves.append(label)
    return subsequences, board_states, next_moves, vocab

def df_to_data(df, fixed_window=False, fixed_window_size=16, sampling_rate=1, algebraic_notation=True, vocab = None):
    """
    Input: Dataframe of training data in which each row represents a full game played between players
    Output: List in which each item represents some game's history up until a particular move, List in the same order in which the associated label is the following move
    """
    vocab = vocab
    board_states, fens, subsequences, next_moves = [], [], [], []
    if vocab is None:
        vocab = VocabularyWithCLS()
    chess_board = chess.Board()
    for game_board, game_moves in zip(df["board"], df["moves"]):
        moves = game_moves.split()
        boards = game_board.split("*")
        # Encode the moves into SAN notation and then into corresponding indices
        encoded_moves = [1]
        for move in moves:
            # Create a move object from the coordinate notation
            move_obj = chess.Move.from_uci(move)
            # There are some broken moves in the data -> stop reading if so
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
                    vocab.add_move(move)
                    encoded_move = vocab.get_id(move)
                    encoded_moves.append(encoded_move)
        chess_board.reset()
        # at this point, encoded moves is [1,2,23,5,...]
        # boards = boards[: len(encoded_moves)-1]
        # Now generate X,Y with sampling
        for i in range(0,len(encoded_moves)-1):
            # TODO: Figure out how to deal with black orientation 'seeing' a different board
            if random.uniform(0, 1) <= sampling_rate and "w" in boards[i]:
                # Board
                board_states.append(fen_to_array_two(boards[i].split(" ")[0]))
                fens.append(boards[i])
                # Sequence of Moves
                subseq = encoded_moves[0 : i + 1]
                if fixed_window and len(subseq) > fixed_window_size:
                    subseq = subseq[-fixed_window_size:]
                subsequences.append(subseq)
                # Label
                label = encoded_moves[i+1]
                next_moves.append(label)
    return subsequences, fens, board_states, next_moves, vocab

def df_to_data_black(df, fixed_window=False, fixed_window_size=16, sampling_rate=1, algebraic_notation=True, vocab = None):
    """
    Input: Dataframe of training data in which each row represents a full game played between players
    Output: List in which each item represents some game's history up until a particular move, List in the same order in which the associated label is the following move
    """
    vocab = vocab
    board_states, fens, subsequences, next_moves = [], [], [], []
    if vocab is None:
        vocab = VocabularyWithCLS()
    chess_board = chess.Board()
    for game_board, game_moves in zip(df["board"], df["moves"]):
        moves = game_moves.split()
        boards = game_board.split("*")
        # Encode the moves into SAN notation and then into corresponding indices
        encoded_moves = [1]
        for move in moves:
            # Create a move object from the coordinate notation
            move_obj = chess.Move.from_uci(move)
            # There are some broken moves in the data -> stop reading if so
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
                    vocab.add_move(move)
                    encoded_move = vocab.get_id(move)
                    encoded_moves.append(encoded_move)
        chess_board.reset()
        # at this point, encoded moves is [1,2,23,5,...]
        # boards = boards[: len(encoded_moves)-1]
        # Now generate X,Y with sampling
        for i in range(len(encoded_moves)-1):
            # TODO: Figure out how to deal with black orientation 'seeing' a different board
            if random.uniform(0, 1) <= sampling_rate and "b" in boards[i].split(" ")[1]:
                # Board
                board_states.append(fen_to_array_two(boards[i].split(" ")[0]))
                fens.append(boards[i])
                # Sequence of Moves
                subseq = encoded_moves[0 : i + 1]
                if fixed_window and len(subseq) > fixed_window_size:
                    subseq = subseq[-fixed_window_size:]
                subsequences.append(subseq)
                # Label
                label = encoded_moves[i+1]
                next_moves.append(label)
    return subsequences, fens, board_states, next_moves, vocab


def df_to_data_fen_only(df, fixed_window=False, fixed_window_size=8, sampling_rate=1, algebraic_notation=True, vocab = None):
    """
    Converts a dataframe of chess games into a dataset for a transformer model.
    Each item in the output list represents the history of a game up to a particular move,
    followed by the CLS token, with moves and FENs separated by special tokens.
    """
    if vocab is None:
        vocab = VocabularyForTransformer()
    vocab.add_move(' ')
    subsequences, next_moves = [], []
    chess_board = chess.Board()
    for game_board, game_moves in zip(df["board"], df["moves"]):
        moves = game_moves.split()
        boards = game_board.split("*")
        sequence = [1]
        for move, board in zip(moves, boards):
            move_obj = chess.Move.from_uci(move)
            if move_obj not in chess_board.legal_moves:
                break
            algebraic_move = chess_board.san(move_obj) if algebraic_notation else move
            chess_board.push(move_obj)
            for char in algebraic_move:
                vocab.add_move(char)
            # Add to vocabulary and sequence
            expanded_fen = expand_fen(board.strip())
            for char in expanded_fen:
                vocab.add_move(char)
            sequence.extend([vocab.get_id("<BOARD>")])
            sequence.extend([vocab.get_id(char) for char in expanded_fen])
            sequence.extend([vocab.get_id("<MOVE>")])
            sequence.extend([vocab.get_id(char) for char in algebraic_move])
            sequence.extend([vocab.get_id("<SEP>")])
        boards = boards[: len(sequence)-1]
        chess_board.reset()
        move_indices = [i for i, token in enumerate(sequence) if token == vocab.get_id('<MOVE>')]
        sep_indices = [i for i, token in enumerate(sequence) if token == vocab.get_id('<SEP>')]
        assert(len(move_indices) == len(sep_indices))

        # Iterate through the game-sequence to create sub-subsequences
        for i in range(len(move_indices)):
            # Check if sampling this subsequence based on the sampling rate
            if random.uniform(0, 1) <= sampling_rate:
                # Sequence
                if fixed_window and i > fixed_window_size:
                    subsequence = sequence[sep_indices[i-fixed_window_size]:move_indices[i]+1]
                else:
                    subsequence = sequence[:move_indices[i]+1]
                subsequences.append(subsequence)
                # Label
                label = sequence[move_indices[i]+1:sep_indices[i]+1]
                next_moves.append(label)
            
    return subsequences, next_moves, vocab

def df_to_data_fen_only_padded(df, fixed_window=False, fixed_window_size=8, sampling_rate=1, algebraic_notation=True, vocab=None):
    if vocab is None:
        vocab = VocabularyForTransformer()
    vocab.add_move(' ')
    subsequences, next_moves = [],[]
    chess_board = chess.Board()
    
    # Constants for padding
    SUBSEQUENCE_PAD_LENGTH = 750
    LABEL_PAD_LENGTH = 7
    pad_id = vocab.get_id('<PAD>')  # Assuming you have a PAD token in your vocabulary

    for game_board, game_moves in zip(df["board"], df["moves"]):
        moves = game_moves.split()
        boards = game_board.split("*")
        sequence = [1]  # Starting token, adjust as necessary
        for move, board in zip(moves, boards):
            move_obj = chess.Move.from_uci(move)
            if move_obj not in chess_board.legal_moves:
                break
            algebraic_move = chess_board.san(move_obj) if algebraic_notation else move
            chess_board.push(move_obj)
            # Process the algebraic move and board for vocabulary
            temp1, temp2 = [],[]
            for char in algebraic_move:
                vocab.add_move(char)
                temp1.append(vocab.get_id(char))
            expanded_fen = expand_fen(board.strip())
            for char in expanded_fen:
                vocab.add_move(char)
                temp2.append(vocab.get_id(char))
            # Append tokens to sequence
            sequence.extend([
                vocab.get_id("<BOARD>"),
                *temp2,
                vocab.get_id("<MOVE>"),
                *temp1,
                vocab.get_id("<SEP>")
            ])
        
        chess_board.reset()

        move_indices = [i for i, token in enumerate(sequence) if token == vocab.get_id('<MOVE>')]
        sep_indices = [i for i, token in enumerate(sequence) if token == vocab.get_id('<SEP>')]

        for i, (move_index, sep_index) in enumerate(zip(move_indices, sep_indices)):
            if random.uniform(0, 1) <= sampling_rate:
                start_index = sep_indices[i - fixed_window_size] if fixed_window and i >= fixed_window_size else 0
                subsequence = sequence[start_index:move_index + 1]
                label = sequence[move_index + 1:sep_index]
                
                # Pad subsequence and label
                subsequence += [pad_id] * (SUBSEQUENCE_PAD_LENGTH - len(subsequence))
                label += [pad_id] * (LABEL_PAD_LENGTH - len(label))
                
                assert len(subsequence) == SUBSEQUENCE_PAD_LENGTH, f"Subsequence length is {len(subsequence)}, expected {SUBSEQUENCE_PAD_LENGTH}"
                if len(label) != LABEL_PAD_LENGTH:
                    print([vocab.get_word(i) for i in label])
                assert len(label) == LABEL_PAD_LENGTH, f"Label length is {len(label)}, expected {LABEL_PAD_LENGTH}"

                subsequences.append(subsequence)
                next_moves.append(label)  # Ensure label is exactly 7 tokens long

    return subsequences, next_moves, vocab


# Function to calculate top-3 accuracy
def top_3_accuracy(y_true, y_pred):
    top3 = torch.topk(y_pred, 3, dim=1).indices
    correct = top3.eq(y_true.view(-1, 1).expand_as(top3))
    return correct.any(dim=1).float().mean().item()

def expand_fen(fen):
    """
    Expands a FEN string into a fixed-length format by expanding the shorthand notations.
    
    Args:
        fen (str): The FEN string representing a chess position.
    
    Returns:
        str: The expanded FEN string with a fixed length for the board representation.
    """
    # Split the FEN string into its main components
    parts = fen.split(' ')
    board, side_to_move, castling, en_passant, halfmove_clock, fullmove_number = parts
    
    # Expand the board shorthand notations
    expanded_board = ''
    rows = board.split('/')
    for row in rows:
        expanded_row = ''
        for char in row:
            if char.isdigit():
                # Replace a digit with that many placeholders ('1' becomes ' ')
                expanded_row += '*' * int(char)
            else:
                expanded_row += char
        expanded_board += expanded_row + '/'
    
    # Remove the trailing slash from the expanded board representation
    expanded_board = expanded_board.rstrip('/')
    
    # Reassemble the FEN string with the expanded board
    expanded_fen = ' '.join([expanded_board, side_to_move, castling, en_passant, halfmove_clock, fullmove_number])
    
    return expanded_fen


# Function to pad move sequences & get their sequence lengths
def pad_sequences(sequences, max_len=None, pad_id=0):
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    padded_sequences = np.full((len(sequences), max_len), pad_id, dtype=int)
    sequence_lengths = np.zeros(len(sequences), dtype=int)
    for i, seq in enumerate(sequences):
        length = len(seq)
        padded_sequences[i, :length] = seq[:length]
        sequence_lengths[i] = length
    return padded_sequences, sequence_lengths

""" Analysis """
def is_legal_move(chess_board, move_san):
    try:
        chess_move = chess_board.parse_san(move_san)
        return chess_move in chess_board.legal_moves
    except ValueError:
        # This handles cases where the SAN move cannot be parsed or is not legal
        return False


def load_board_state_from_san(moves, vocab):
    board = chess.Board()
    count = 0
    for index in moves:
        try:
            if index == 1:
                continue
            if index == 0 and count > 2 :
                return board
            else:
                move_san = vocab.get_move(index.item())
                move = board.parse_san(move_san)
                board.push(move)
                count += 1
        except ValueError:
            # Handle invalid moves, e.g., break the loop or log an error
            break
    return board


def convert_to_grayscale(image):
    # Assuming 'image' is a numpy array of shape (12, height, width)
    # Convert to grayscale by averaging the channels

    grayscale_image = image.mean(axis=0).mean(axis=0)

    return grayscale_image

def visualize_heatmap_on_chessboard(chessboard, heatmap):


    # Convert the 12-channel chessboard to a grayscale image for visualization
    grayscale_chessboard = convert_to_grayscale(chessboard)
    # Normalize the grayscale chessboard for visualization
    normalized_chessboard = grayscale_chessboard / grayscale_chessboard.max()

    zeros_channel = np.zeros_like(normalized_chessboard)
    
    three_channel_chessboard = np.stack((normalized_chessboard,zeros_channel,zeros_channel), axis=-1)

    # Ensure the heatmap is in the correct format and resize it to match the chessboard image
    heatmap_resized = np.resize(heatmap, (grayscale_chessboard.shape[0], grayscale_chessboard.shape[1]))
    heatmap_resized = np.stack((heatmap_resized,zeros_channel,zeros_channel), axis=-1)
    # Overlay the heatmap on the grayscale chessboard
    visualization = show_cam_on_image(three_channel_chessboard, heatmap_resized, use_rgb=True)

    # Display the visualization
    plt.imshow(visualization, cmap='hot')
    plt.colorbar()
    plt.show()

def show_maps_on_training_data(vocab, model, train_loader, num_samples = 10, conv2 = True, multimodal = False):
    class ModelWrapper(nn.Module):
        def __init__(self, original_model):
            super(ModelWrapper, self).__init__()
            self.original_model = original_model

        def forward(self, combined_input):
            # Split the combined_input tensor into the expected inputs for the original model
            input1, input2, input3 = self.split_inputs(combined_input)
            # Forward these inputs through the original model
            return self.original_model(input1, input2, input3)

        def split_inputs(self, combined_input):
            # Assuming 'combined_input' was concatenated along a new last dimension
            # Adjust the slicing based on how you concatenated the tensors
            input1 = combined_input[:, :, :, :, 0]  # Shape: [1, 12, 8, 8]
            input2 = combined_input[:, 0, 0, 0, 1]  # Shape: [1, 16], needs further reshaping
            input3 = combined_input[0, 0, 0, 0, 2]  # Scalar, needs further reshaping

            # Reshape 'input2' and 'input3' to their original shapes
            input2 = input2.view(1, -1)  # Assuming the original shape was [1, 16]
            input3 = input3.view(1)     # Assuming the original shape was [1]

            return input1, input2, input3

    # Mapping from tensor indices to chess pieces
    index_to_piece = {
        0: chess.PAWN, 1: chess.ROOK, 2: chess.KNIGHT,
        3: chess.BISHOP, 4: chess.QUEEN, 5: chess.KING,
        6: chess.PAWN, 7: chess.ROOK, 8: chess.KNIGHT,
        9: chess.BISHOP, 10: chess.QUEEN, 11: chess.KING
    }

    model.eval()
    # Assuming ChessCNN_no_pooling is your model class
    if conv2:
        target_layers = [model.conv2]  # Use the last conv layer
    else: 
        target_layers = [model.fc]
    # Initialize GradCAM with the model and target layers
    cam = GradCAM(model=model, target_layers=target_layers)

    # multimodal part is.... gg
    i = 0
    if multimodal:
        cam = GradCAM(model=ModelWrapper(model), target_layers=target_layers)
        for boards, sequences, lengths, labels in train_loader:  # Iterate over your data
            # Specify the target; if None, the highest scoring category will be used
            # For simplicity, we're using None here
            if i > 0 and i < num_samples:
                boards, sequences, lengths, labels = boards.to(device, non_blocking = True), sequences.to(device, non_blocking = True), lengths, labels.to(device, non_blocking = True)
                _, predicted = torch.max(model(boards, sequences, lengths).data, 1)
                print(model(boards, sequences, lengths).data.shape)
                print(vocab.get_move(predicted.item()))
                print(vocab.get_move(labels.item()))
                targets = None
                # Expand dimensions of 'sequences' and 'lengths' to match 'boards'
                sequences_expanded = sequences.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Now it's [1, 16, 1, 1, 1]
                lengths_expanded = lengths.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Now it's [1, 1, 1, 1, 1]
                print(boards.unsqueeze(-1).shape)
                print(sequences_expanded.shape)
                print(lengths_expanded.shape)
                # Concatenate along a new dimension
                combined_input = torch.cat((boards.unsqueeze(-1), sequences_expanded, lengths_expanded.to(device, non_blocking = True)), dim=-1)

                # Generate the CAM mask
                grayscale_cam = cam(input_tensor=combined_input, targets=targets)
                # Visualize the first image in the batch
                visualize_heatmap_on_chessboard(boards, grayscale_cam)

                # This is 
                board = chess.Board(None)  # Start with an empty board
                image = boards.sum(axis=0)
                # Iterate over the tensor to place pieces on the board
                for channel, piece_type in index_to_piece.items():
                    for row in range(8):
                        for col in range(8):
                            if image[channel, row, col] > 0:  # Assuming nonzero value indicates presence of a piece
                                piece_color = chess.WHITE if channel < 6 else chess.BLACK
                                piece = chess.Piece(piece_type, piece_color)
                                square = chess.square(col, 7-row)  # chess.square() needs file index (0-7) and rank index (0-7)
                                board.set_piece_at(square, piece)

                # Now, 'board' contains the chessboard representation. You can print it as text:
                print(board)
                print("\n---------------------\n")
            elif i > num_samples:
                break
            i += 1
    else:
        for image, labels in train_loader:  # Iterate over your data
            # Specify the target; if None, the highest scoring category will be used
            # For simplicity, we're using None here
            if i > 0 and i < num_samples:
                _, predicted = torch.max(model(image).data, 1)
                print(vocab.get_move(predicted.item()))
                print(vocab.get_move(labels.item()))
                targets = None
                # Generate the CAM mask
                grayscale_cam = cam(input_tensor=image, targets=targets)
                # Visualize the first image in the batch
                visualize_heatmap_on_chessboard(image, grayscale_cam)

                # This is 
                board = chess.Board(None)  # Start with an empty board
                image = image.sum(axis=0)
                # Iterate over the tensor to place pieces on the board
                for channel, piece_type in index_to_piece.items():
                    for row in range(8):
                        for col in range(8):
                            if image[channel, row, col] > 0:  # Assuming nonzero value indicates presence of a piece
                                piece_color = chess.WHITE if channel < 6 else chess.BLACK
                                piece = chess.Piece(piece_type, piece_color)
                                square = chess.square(col, 7-row)  # chess.square() needs file index (0-7) and rank index (0-7)
                                board.set_piece_at(square, piece)

                # Now, 'board' contains the chessboard representation. You can print it as text:
                print(board)
                print("\n---------------------\n")
            elif i > num_samples:
                break
            i += 1