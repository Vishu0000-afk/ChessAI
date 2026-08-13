"""Board <-> neural-network tensor encoding.

Positions are encoded as `NUM_CHANNELS x 8 x 8` float planes (White's
point of view, row 0 = rank 8, matching python-chess square numbering):

    0..5     White pieces (P,N,B,R,Q,K)
    6..11    Black pieces (P,N,B,R,Q,K)
    12       side to move (1.0 if White)
    13       en passant target square
    14       White kingside  castling right
    15       White queenside castling right
    16       Black kingside  castling right
    17       Black queenside castling right

The policy space is 64x64 = 4096 (from-square * 64 + to-square), the same
mapping used by the AlphaZero-style policy head.
"""

from __future__ import annotations

from typing import List

import chess
import numpy as np

NUM_CHANNELS = 18
NUM_SQUARES = 64
POLICY_SIZE = NUM_SQUARES * NUM_SQUARES

_PIECE_PLANE: dict = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def move_to_index(move: chess.Move) -> int:
    """Map a move to a policy index (from_square * 64 + to_square)."""
    return move.from_square * NUM_SQUARES + move.to_square


def index_to_move(index: int) -> chess.Move:
    """Map a policy index back to a from/to square pair (promotion = queen)."""
    from_square = index // NUM_SQUARES
    to_square = index % NUM_SQUARES
    return chess.Move(from_square, to_square)


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a position into an (NUM_CHANNELS, 8, 8) float32 array."""
    planes = np.zeros((NUM_CHANNELS, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        channel = _PIECE_PLANE[piece.piece_type]
        if piece.color == chess.BLACK:
            channel += 6
        planes[channel, rank, file] = 1.0

    if board.turn == chess.WHITE:
        planes[12] = 1.0

    ep = board.ep_square
    if ep is not None:
        planes[13, chess.square_rank(ep), chess.square_file(ep)] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        planes[14] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[15] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[16] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[17] = 1.0

    return planes


def encode_board_batch(boards: List[chess.Board]) -> np.ndarray:
    """Encode many boards into a single (N, NUM_CHANNELS, 8, 8) float32 array."""
    arr = np.empty((len(boards), NUM_CHANNELS, 8, 8), dtype=np.float32)
    for i, board in enumerate(boards):
        arr[i] = encode_board(board)
    return arr


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Return a boolean (POLICY_SIZE,) mask of the legal moves in a position."""
    mask = np.zeros(POLICY_SIZE, dtype=np.bool_)
    for move in board.legal_moves:
        mask[move_to_index(move)] = True
    return mask


def pack_mask(mask: np.ndarray) -> np.ndarray:
    """Pack a (POLICY_SIZE,) bool mask into 64 uint64 words (512 bytes)."""
    packed = np.zeros(64, dtype=np.uint64)
    for i in range(64):
        chunk = mask[i * 64:(i + 1) * 64]
        word = 0
        for j in range(64):
            if chunk[j]:
                word |= np.uint64(1) << np.uint64(j)
        packed[i] = word
    return packed


def unpack_mask(packed: np.ndarray) -> np.ndarray:
    """Unpack 64 uint64 words back into a (POLICY_SIZE,) bool mask."""
    mask = np.zeros(POLICY_SIZE, dtype=np.bool_)
    for i in range(64):
        word = int(packed[i])
        for j in range(64):
            if (word >> j) & 1:
                mask[i * 64 + j] = True
    return mask