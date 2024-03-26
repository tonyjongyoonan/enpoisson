import torch
import torch.nn
import pickle
import chess
from nn_models import *
from utils import fen_to_array_two, is_legal_move
from typing import List

vocab_path = "vocab.pkl"
model_path = "multimodalmodel-exp-12.pth"


# note: can only be used for white positions
class ChessEngine:
    def __init__(self, model_path):
        self.device = torch.device("cpu")
        with open(vocab_path, "rb") as inp:
            self.vocab = pickle.load(inp)
        self.d_hidden = 256
        self.d_embed = 64
        self.d_out = len(self.vocab.id_to_move.keys())
        self.model = MultiModalSeven(
            self.vocab, self.d_embed, self.d_hidden, self.d_out
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # Set the model to evaluation mode

    @staticmethod
    def pad_last_move_sequence(last_move_sequence_ids: List[int], sequence_length: int):
        return last_move_sequence_ids + [0 for _ in range(16 - sequence_length)]

    def top_k_legal_moves(
        self, fen: str, sorted_move_indices: torch.Tensor, *, top_k: int
    ):
        """
        Finds the top_k legal moves. Uses short-circuit evaluation so it is faster
        than filtering the whole set of possitive moves.
        """
        chess_board = chess.Board(fen)
        output_moves = []
        for move_idx in sorted_move_indices:
            # Convert index to move (e.g., 100 |-> 'e2e4')
            move = self.vocab.get_move(move_idx.item())
            if len(output_moves) >= top_k:
                break
            if is_legal_move(chess_board, move):
                output_moves.append(self.vocab.get_id(move))
        return [self.vocab.id_to_move[move] for move in output_moves]

    def call_model(
        self,
        board: list[list[list[int]]],
        last_move_sequence_ids: list[int],
        sequence_length: int,
    ):
        """
        Does the work of converting the types to tensors in order to use the model.
        Uses a batch size of 1.
        """
        board_tensor = torch.tensor([board], dtype=torch.float32, device=self.device)
        move_sequence_tensor = torch.tensor(
            [
                ChessEngine.pad_last_move_sequence(
                    last_move_sequence_ids, sequence_length
                )
            ],
            dtype=torch.long,
            device=self.device,
        )
        sequence_length_tensor = torch.tensor(
            [sequence_length], dtype=torch.long, device=self.device
        )
        return self.model(
            board_tensor,
            move_sequence_tensor,
            sequence_length_tensor,
        )[
            0
        ]  # batch of size 1

    def get_human_move(self, fen: str, last_move_sequence: List[str], *, top_k: int):
        """
        fen: a fen in string format representing the position
        last_move_sequence: list of last 16 half-moves made
        top_k: keyword argument, number moves to return
        """
        fen_board: str = fen.split()[0]
        board = fen_to_array_two(fen_board)
        sequence_length = len(last_move_sequence)
        last_move_sequence_ids = [
            self.vocab.move_to_id[move] for move in last_move_sequence
        ]
        with torch.no_grad():
            model_output = self.call_model(
                board, last_move_sequence_ids, sequence_length
            )
        # add difficulty bar here
        _, sorted_move_indices = torch.sort(model_output, descending=True)
        # each index contains the probability of the move representing that index
        model_output_decimal = torch.softmax(model_output, dim=0)
        top_k_moves = self.top_k_legal_moves(fen, sorted_move_indices, top_k=top_k)
        return {
            move: model_output_decimal[self.vocab.move_to_id[move]].item()
            for move in top_k_moves
        }


if __name__ == "__main__":
    sample_fen = "r1bqkb1r/ppp2ppp/2n1p3/8/3PpB2/4PN2/PPP2PPP/R2QKB1R w KQkq - 0 7"
    sample_last_16_moves = [
        "d4",
        "d5",
        "Bf4",
        "Nc6",
        "e3",
        "Nf6",
        "Nf3",
        "e6",
        "Nbd2",
        "Ne4",
        "Nxe4",
        "dxe4",
    ]
    engine = ChessEngine(model_path)
    print("model loaded")
    print(engine.get_human_move(sample_fen, sample_last_16_moves, top_k=3))
