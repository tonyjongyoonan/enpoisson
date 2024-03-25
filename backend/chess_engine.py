import torch
import torch.nn
import pickle
import chess
from nn_models import *
from utils import fen_to_array_two, is_legal_move
from typing import List

sample_fen = "rnbqk2r/ppp1ppbp/6p1/3n4/3P1B2/4P3/PP3PPP/RN1QKBNR w KQkq - 0 6"
sample_last_16_moves = [
    "d4",
    "d5",
    "Bf4",
    "Nf6",
    "e3",
    "g6",
    "c4",
    "g7",
    "cxd5",
    "Nxd5",
]
sample_length = len(sample_last_16_moves)
vocab_path = "vocab.pkl"
model_path = "multimodalmodel-exp-12.pth"


# note: can only be used for white positions
class ChessEngine:
    def __init__(self, model_path):
        device = torch.device("cpu")
        with open(vocab_path, "rb") as inp:
            self.vocab = pickle.load(inp)
        self.d_hidden = 256
        self.d_embed = 64
        self.d_out = len(self.vocab.id_to_move.keys())
        self.model = MultiModalSeven(
            self.vocab, self.d_embed, self.d_hidden, self.d_out
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()  # Set the model to evaluation mode

    @staticmethod
    def pad_last_move_sequence(last_move_sequence_ids: List[int], sequence_length: int):
        return last_move_sequence_ids + [0 for _ in range(16 - sequence_length)]

    def get_human_move(
        self, fen: str, last_move_sequence: List[str], sequence_length: int
    ):
        """
        last_move_sequence: list of last 16 moves
        sequence_length: length of the sequence. should be <= 16
        """
        fen_board: str = fen.split()[0]
        board = fen_to_array_two(fen_board)
        last_move_sequence_ids = [
            self.vocab.move_to_id[move] for move in last_move_sequence
        ]
        with torch.no_grad():
            model_output = self.model(
                torch.tensor([board], dtype=torch.float32),
                torch.tensor(
                    [
                        ChessEngine.pad_last_move_sequence(
                            last_move_sequence_ids, sequence_length
                        )
                    ],
                    dtype=torch.long,
                ),
                torch.tensor([sequence_length], dtype=torch.long),
            )[0]
        output_probabilities = torch.softmax(
            model_output,
            dim=0,
        )
        # sorted_probs, sorted_indices = torch.sort(model_output, descending=True)
        # # import chess board from fen
        # chess_board = chess.Board(fen)
        # for move_idx in sorted_indices:
        #     move = self.vocab.get_move(
        #         move_idx.item()
        #     )  # Convert index to move (e.g., 'e2e4')
        #     if is_legal_move(chess_board, move):
        #         predicted_move = self.vocab.get_id(move)
        #         break
        _, best_move = torch.max(model_output, 0)
        return self.vocab.id_to_move[best_move.item()]


if __name__ == "__main__":
    engine = ChessEngine(model_path)
    print(engine.get_human_move(sample_fen, sample_last_16_moves, sample_length))
