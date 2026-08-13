"""Learning package: self-play experience storage and continuous training."""

from src.learning.dataset import (
    ChunkWriter,
    SelfPlayDataset,
    list_chunks,
    read_chunk,
    write_chunk,
)
from src.learning.encoding import encode_board, encode_board_batch, legal_move_mask, move_to_index
from src.learning.learner import Learner
from src.learning.model_manager import CheckpointMetadata, ModelManager
from src.learning.network import ChessNet, create_chess_net
from src.learning.replay_buffer import ReplayBuffer
from src.learning.trainer import Trainer

__all__ = [
    "ChunkWriter",
    "SelfPlayDataset",
    "list_chunks",
    "read_chunk",
    "write_chunk",
    "encode_board",
    "encode_board_batch",
    "legal_move_mask",
    "move_to_index",
    "Learner",
    "CheckpointMetadata",
    "ModelManager",
    "ChessNet",
    "create_chess_net",
    "ReplayBuffer",
    "Trainer",
]