import torch
import matplotlib.pyplot as plt
import chess
import torch.nn as nn
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import chess
import numpy as np
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def top_3_accuracy(y_true, y_pred):
    top3 = torch.topk(y_pred, 3, dim=1).indices
    correct = top3.eq(y_true.view(-1, 1).expand_as(top3))
    return correct.any(dim=1).float().mean().item()

def get_embedding_matrix(vocab, d_embed):
    n_embed = len(vocab.move_to_id)
    return np.random.normal(0, 1, (n_embed, d_embed))

""" Objects """

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

class SequenceDataset(Dataset):
    def __init__(self, sequences, lengths, labels):
        self.sequences = sequences
        self.lengths = lengths
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx], self.labels[idx]
    
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