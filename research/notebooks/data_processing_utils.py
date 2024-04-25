import torch
import chess
import numpy as np
import torch
import chess
import pandas as pd
import random
import numpy as np
import dask.dataframe as dd 

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

def construct_vocab(df, cls = True, algebraic_notation = True, with_checkmate = True):
    if cls: 
        vocab = VocabularyWithCLS()
    else: 
        vocab = Vocabulary()
    board = chess.Board()
    illegal_moves = 0
    for game_moves in df["moves"]:
        moves = game_moves.split()
        for move in moves:
            move_obj = chess.Move.from_uci(move)
            if move_obj in board.legal_moves:
                if algebraic_notation:
                    algebraic_move = board.san(move_obj)
                    board.push(move_obj)
                    if '#' in algebraic_move and (not with_checkmate):
                        algebraic_move = algebraic_move[:-1] + '+'
                    vocab.add_move(algebraic_move)
                else:
                    vocab.add_move(move)
            else:
                illegal_moves += 1
                break
        board.reset()
    return vocab, illegal_moves

def construct_special_vocab(df, algebraic_notation = True):
    """This tokenizes FENs and moves into the same space. Meant for Transformer"""
    vocab = VocabularyForTransformer()
    pass

def df_to_multimodal_memmap(files, vocab, elo, sampling_rate = 0.125):
    trainX_filenames, trainY_filenames, fens_filenames, trainX_sequences_filenames, trainX_seqlengths_filenames, trainY_filenames = [],[],[], [], []
    for file in files:
        df = pd.read_csv(file)
        folder_name = file.split('-')[1].split('.')[0]
        trainX, fens, trainX_sequences, trainY, vocab = df_to_multimodal_data(df, vocab, fixed_window=True, sampling_rate=sampling_rate)
        trainX_sequences, trainX_seqlengths  = pad_sequences(trainX_sequences)
        trainX_filenames.append(save_as_memmap(trainX, f'./../data/{elo}/{folder_name}/trainX.memmap'))
        trainY_filenames.append(save_as_memmap(trainY, f'./../data/{elo}/{folder_name}/trainY.memmap'))
        trainX_seqlengths_filenames.append(save_as_memmap(trainX_seqlengths, f'./../data/{elo}/{folder_name}/trainX_seqlengths.memmap'))
        trainX_sequences_filenames.append(save_as_memmap(trainX_sequences, f'./../data/{elo}/{folder_name}/trainX_sequences.memmap'))

        df = pd.DataFrame(fens, columns=['fens'])
        csv_filename = f'./../data/{elo}/{folder_name}/fens.csv'
        df.to_csv(csv_filename, index=False)
    pass 

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

def df_to_data_board_only(df, vocab, sampling_rate=1.0, algebraic_notation=True):
    """
    Input: Dataframe of training data in which each row represents a full game played between players
    Output: List in which each item represents some game's history up until a particular move, List in the same order in which the associated label is the following move
    """
    if vocab is None: 
        vocab = VocabularyWithCLS()
    board_states = []
    next_moves = []
    fens = []
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

def df_to_multimodal_data(df, vocab, fixed_window=False, fixed_window_size=16, sampling_rate=1, algebraic_notation=True):
    """
    Input: Dataframe of training data in which each row represents a full game played between players
    Output: List in which each item represents some game's history up until a particular move, List in the same order in which the associated label is the following move
    """
    board_states, fens, subsequences, next_moves = [], [], [], []
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
    return np.asarray(subsequences), fens, np.asarray(board_states,dtype=np.bool), np.asarray(next_moves), vocab

def df_to_data_board_only_np(df, vocab, sampling_rate=1.0, algebraic_notation=True):
    if vocab is None: 
        vocab = VocabularyWithCLS()
    initial_size = 1024  # Initial size of the arrays
    resize_factor = 2    # Factor by which to resize the arrays if needed
    board_states = np.empty((initial_size, 8, 8, 12), dtype=np.bool_)
    next_moves = np.empty(initial_size, dtype=int)
    fens = []
    chess_board = chess.Board()
    idx = 0  # Current index to insert data
    
    for game_board, game_moves in zip(df["board"], df["moves"]):
        moves = game_moves.split()
        boards = game_board.split("*")
        encoded_moves = []
        for move in moves:
            move_obj = chess.Move.from_uci(move)
            if algebraic_notation and (move_obj not in chess_board.legal_moves):
                break
            else:
                if algebraic_notation:
                    algebraic_move = chess_board.san(move_obj)
                    chess_board.push(move_obj)
                    vocab.add_move(algebraic_move)
                    encoded_move = vocab.get_id(algebraic_move)
                else:
                    vocab.add_move(move)
                    encoded_move = vocab.get_id(move)
                encoded_moves.append(encoded_move)
        if algebraic_notation:
            chess_board.reset()
        
        boards = boards[:len(encoded_moves)]
        for i in range(len(encoded_moves) - 1):
            if random.uniform(0, 1) <= sampling_rate:
                label = encoded_moves[i + 1]
                fens.append(boards[i])
                if idx >= board_states.shape[0]:  # Check if resize is needed
                    new_size = board_states.shape[0] * resize_factor
                    board_states = np.resize(board_states, (new_size, 8, 8, 12))
                    next_moves = np.resize(next_moves, new_size)
                board_states[idx] = fen_to_array_three(boards[i])
                next_moves[idx] = label
                idx += 1

    # Efficiently resize to the final size by creating new arrays
    board_states_final = np.array(board_states[:idx], dtype=np.bool_)
    next_moves_final = np.array(next_moves[:idx], dtype=int)

    del chess_board, board_states, next_moves  # Clean up original arrays
    return board_states_final, fens, next_moves_final, vocab

def df_to_multimodal_data_color(df, vocab, color = 'w', fixed_window=False, fixed_window_size=16, sampling_rate=1, algebraic_notation=True):
    """
    Input: Dataframe of training data in which each row represents a full game played between players
    Output: List in which each item represents some game's history up until a particular move, List in the same order in which the associated label is the following move
    """
    if vocab is None: 
        vocab = VocabularyWithCLS()

    board_states, fens, subsequences, next_moves = [], [], [], []
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
            if random.uniform(0, 1) <= sampling_rate and (color in boards[i].split(" ")[1]):
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
        del chess_board
    return np.asarray(subsequences), fens, np.asarray(board_states,dtype=np.bool_), next_moves, vocab

def df_to_data_fen_only(df, vocab, fixed_window=False, fixed_window_size=8, sampling_rate=1, algebraic_notation=True):
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

    """Turning Data into Memmaps
    """
    
def save_as_memmap(array, filename):
    # Determine the dtype and shape of the array to create a compatible memmap
    dtype = array.dtype
    shape = array.shape
    
    # Create a memmap file with write mode, which will also allocate the disk space
    memmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
    
    # Copy the data into the memmap array
    memmap_array[:] = array[:]
    
    # Flush memory changes to disk to ensure all data is written
    memmap_array.flush()

    # Return the path for confirmation
    return filename

# If you forget the shape
def find_working_shape(filename, dtype, max_first_dim, other_dims):

    # Try decreasing sizes from the max_first_dim until we find a working shape
    for first_dim in range(max_first_dim, 0, -1):
        shape_to_try = (first_dim,) + other_dims
        
        try:
            # Attempt to load the memmap with the current shape
            memmap_array = np.memmap(filename, dtype=dtype, mode='r', shape=shape_to_try)
            # If successful, return the array
            print(f"Successful shape: {shape_to_try}")
            return memmap_array
        except ValueError as e:
            # Catch the ValueError if the shape is not feasible, and try the next size
            continue
    
    raise ValueError("Could not find a working shape within the given bounds.")

# Function to load a memmap file
def load_memmap(filename, dtype, shape):
    # Load the memmap file with read-only mode
    return np.memmap(filename, dtype=dtype, mode='r', shape=shape)