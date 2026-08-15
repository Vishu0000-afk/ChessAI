"""Board <-> neural-network tensor encoding.

Positions are encoded from the side-to-move's perspective (AlphaZero-style).
The board is always rendered with the side to move at the bottom: when Black
is to move the board is rotated 180 degrees and the piece colors are swapped,
so a color-swapped, rotated position produces an identical tensor. This makes
the network color-invariant by construction.

Planes are `NUM_CHANNELS x 8 x 8` float (row 0 = rank 8, matching
python-chess square numbering):

    0..5     pieces of the side to move (P,N,B,R,Q,K)
    6..11    opponent pieces (P,N,B,R,Q,K)
    12       side to move (always 1.0 after the color-relative transform)
    13       en passant target square
    14       our kingside  castling right
    15       our queenside castling right
    16       opponent kingside  castling right
    17       opponent queenside castling right

The policy space is 64x64 = 4096 (from-square * 64 + to-square) in the same
side-to-move-relative frame: move indices are mirrored when it is Black's turn.
"""

from __future__ import annotations

from typing import List

import chess
import numpy as np

NUM_CHANNELS = 18
NUM_SQUARES = 64
POLICY_SIZE = NUM_SQUARES * NUM_SQUARES


def mirror_square(square: int) -> int:
    """180-degree rotation (rank + file flip): the color-relative transform."""
    return 63 - square


def move_to_index(move: chess.Move, stm: int = chess.WHITE) -> int:
    """Map a move to a policy index in the side-to-move-relative frame."""
    from_square = mirror_square(move.from_square) if stm == chess.BLACK else move.from_square
    to_square = mirror_square(move.to_square) if stm == chess.BLACK else move.to_square
    return from_square * NUM_SQUARES + to_square


def move_to_indices(moves, stm: int = chess.WHITE) -> np.ndarray:
    """Vectorized :func:`move_to_index` over a list of moves.

    Returns an (N,) int64 array of policy indices in the side-to-move-relative
    frame, matching the order of ``moves``.
    """
    from_square = np.fromiter(
        (m.from_square for m in moves), dtype=np.int64, count=len(moves)
    )
    to_square = np.fromiter(
        (m.to_square for m in moves), dtype=np.int64, count=len(moves)
    )
    if stm == chess.BLACK:
        from_square = 63 - from_square
        to_square = 63 - to_square
    return from_square * NUM_SQUARES + to_square


def index_to_move(index: int, stm: int = chess.WHITE) -> chess.Move:
    """Map a policy index back to a from/to square pair (promotion = queen)."""
    from_square = index // NUM_SQUARES
    to_square = index % NUM_SQUARES
    if stm == chess.BLACK:
        from_square = mirror_square(from_square)
        to_square = mirror_square(to_square)
    return chess.Move(from_square, to_square)


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a position into an (NUM_CHANNELS, 8, 8) float32 array.

    The result is expressed from the side to move's perspective: the side to
    move is treated as "White" (planes 0..5), opponent pieces go to 6..11,
    and all squares/castling/ep data are mirrored when Black is to move.
    """
    planes = np.zeros(NUM_CHANNELS * NUM_SQUARES, dtype=np.float32)
    stm_white = board.turn == chess.WHITE
    pm = board.piece_map()
    squares = np.fromiter(pm.keys(), dtype=np.int64, count=len(pm))
    pieces = np.fromiter((p.piece_type for p in pm.values()), dtype=np.int64, count=len(pm))
    colors = np.fromiter((p.color for p in pm.values()), dtype=np.int64, count=len(pm))

    if stm_white:
        channels = (pieces - 1) + 6 * (1 - colors)
    else:
        channels = (pieces - 1) + 6 * colors
        squares = 63 - squares
    # channel*64 + square == flat (channel, rank, file) index (C-order).
    planes[channels * NUM_SQUARES + squares] = 1.0
    planes = planes.reshape(NUM_CHANNELS, 8, 8)

    # The side to move is always "White" in the relative frame.
    planes[12] = 1.0

    ep = board.ep_square
    if ep is not None:
        if not stm_white:
            ep = mirror_square(ep)
        planes[13, chess.square_rank(ep), chess.square_file(ep)] = 1.0

    # Castling rights: rotate colors and sides when Black is to move
    # (our kingside <-> opponent queenside, our queenside <-> opponent kingside).
    wk = board.has_kingside_castling_rights(chess.WHITE)
    wq = board.has_queenside_castling_rights(chess.WHITE)
    bk = board.has_kingside_castling_rights(chess.BLACK)
    bq = board.has_queenside_castling_rights(chess.BLACK)
    if stm_white:
        planes[14] = float(wk)
        planes[15] = float(wq)
        planes[16] = float(bk)
        planes[17] = float(bq)
    else:
        planes[14] = float(bq)
        planes[15] = float(bk)
        planes[16] = float(wq)
        planes[17] = float(wk)

    return planes


def encode_board_batch(boards: List[chess.Board]) -> np.ndarray:
    """Encode many boards into a single (N, NUM_CHANNELS, 8, 8) float32 array."""
    arr = np.empty((len(boards), NUM_CHANNELS, 8, 8), dtype=np.float32)
    for i, board in enumerate(boards):
        arr[i] = encode_board(board)
    return arr


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Return a boolean (POLICY_SIZE,) mask of the legal moves in a position.

    The mask is expressed in the same side-to-move-relative policy frame as
    the network's policy head, so it is consistent with ``move_to_index``.
    """
    moves = list(board.legal_moves)
    mask = np.zeros(POLICY_SIZE, dtype=np.bool_)
    if moves:
        mask[move_to_indices(moves, board.turn)] = True
    return mask


def pack_mask(mask: np.ndarray) -> np.ndarray:
    """Pack a (POLICY_SIZE,) bool mask into 64 uint64 words (512 bytes).

    Bit ``j`` of word ``i`` holds ``mask[i * 64 + j]``. Vectorized: expands to
    a (…, 64, 64) boolean cube, scales by power-of-two columns, and sums.
    """
    single = mask.ndim == 1
    m = mask[None, :] if single else mask
    n = m.shape[0]
    powers = np.uint64(1) << np.arange(64, dtype=np.uint64)
    words = (m.reshape(n, 64, 64) * powers).sum(axis=2).astype(np.uint64)
    return words[0] if single else words


def unpack_mask(packed: np.ndarray) -> np.ndarray:
    """Unpack 64 uint64 words back into a (POLICY_SIZE,) bool mask."""
    mask = np.zeros(POLICY_SIZE, dtype=np.bool_)
    for i in range(64):
        word = int(packed[i])
        for j in range(64):
            if (word >> j) & 1:
                mask[i * 64 + j] = True
    return mask


# =============================================================================
# Mirror augmentation (horizontal file mirror of the relative frame)
# =============================================================================
def mirror_file_square(square: int) -> int:
    """Horizontal (left-right) mirror: keeps the rank, flips the file."""
    return (square // 8) * 8 + (7 - (square % 8))


def mirror_move_index(index: int) -> int:
    """Mirror a relative policy index horizontally (from/to files flip)."""
    from_square = index // NUM_SQUARES
    to_square = index % NUM_SQUARES
    return mirror_file_square(from_square) * NUM_SQUARES + mirror_file_square(to_square)


def mirror_move_indices(indices: np.ndarray) -> np.ndarray:
    """Vectorized :func:`mirror_move_index` for an (N,) int array."""
    from_square = indices // NUM_SQUARES
    to_square = indices % NUM_SQUARES
    return mirror_file_square(from_square) * NUM_SQUARES + mirror_file_square(to_square)


def mirror_planes(planes: np.ndarray) -> np.ndarray:
    """Horizontally mirror encoded planes (flips the file axis)."""
    return np.flip(planes, axis=-1)


def mirror_mask_packed(packed: np.ndarray) -> np.ndarray:
    """Mirror a packed legal mask (single (64,) or batch (N, 64)).

    Fully vectorized: unpacks to (…, 4096) bools, permutes the policy indices,
    and re-packs into uint64 words.
    """
    single = packed.ndim == 1
    if single:
        packed = packed[None, :]
    n = packed.shape[0]
    bits = np.arange(64, dtype=np.uint64)
    mask = ((packed.reshape(n, 64, 1) >> bits) & 1).reshape(n, POLICY_SIZE).astype(bool)

    perm = mirror_move_indices(np.arange(POLICY_SIZE, dtype=np.int64))
    mirrored = np.zeros_like(mask)
    mirrored[:, perm] = mask

    powers = np.uint64(1) << np.arange(64, dtype=np.uint64)
    out = (mirrored.reshape(n, 64, 64) * powers).sum(axis=2).astype(np.uint64)
    return out[0] if single else out