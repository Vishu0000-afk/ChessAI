"""Tests for the learning package (encoding, replay buffer, dataset,
trainer, model manager)."""

import numpy as np
import pytest
import torch

from src.learning.dataset import (
    ChunkWriter,
    SelfPlayDataset,
    list_chunks,
    read_chunk,
    write_chunk,
)
from src.learning.encoding import (
    NUM_CHANNELS,
    POLICY_SIZE,
    encode_board,
    encode_board_batch,
    legal_move_mask,
    move_to_index,
    pack_mask,
    unpack_mask,
)
from src.learning.model_manager import CheckpointMetadata, ModelManager
from src.learning.network import create_chess_net
from src.learning.replay_buffer import ReplayBuffer
from src.learning.trainer import Trainer
from src.learning.learner import Learner
import chess


# ----------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------
def test_encode_board_shape_and_channel_count():
    board = chess.Board()
    x = encode_board(board)
    assert x.shape == (NUM_CHANNELS, 8, 8)
    assert x.dtype == np.float32
    batch = encode_board_batch([board, board])
    assert batch.shape == (2, NUM_CHANNELS, 8, 8)


def test_move_index_round_trip():
    board = chess.Board()
    for move in list(board.legal_moves)[:5]:
        assert move.from_square * 64 + move.to_square == move_to_index(move)


def test_legal_mask_round_trip():
    board = chess.Board()
    mask = legal_move_mask(board)
    assert mask.shape == (POLICY_SIZE,)
    assert mask.sum() == 20  # 20 legal moves from the start position
    packed = pack_mask(mask)
    assert packed.shape == (64,)
    np.testing.assert_array_equal(unpack_mask(packed), mask)


def test_legal_mask_promotion_position():
    board = chess.Board("8/P7/8/8/8/8/8/k1K5 w - - 0 1")
    mask = legal_move_mask(board)
    # a7-a8 promotions (queen/rook/bishop/knight) + a7a6.
    promo = [m for m in board.legal_moves if m.to_square == chess.A8]
    assert len(promo) == 4
    for m in promo:
        assert mask[move_to_index(m)]


# ----------------------------------------------------------------------
# Replay buffer
# ----------------------------------------------------------------------
def _sample(board=None, version=1):
    board = board or chess.Board()
    moves = list(board.legal_moves)
    move = moves[0]
    return {
        "input": encode_board(board),
        "move_index": move_to_index(move),
        "legal_packed": pack_mask(legal_move_mask(board)),
        "value": 1.0,
        "version": version,
    }


def test_replay_buffer_extend_and_sample():
    buf = ReplayBuffer(capacity=100)
    buf.extend([_sample(version=i) for i in range(10)])
    assert len(buf) == 10
    batch = buf.sample(5)
    assert batch["input"].shape == (5, NUM_CHANNELS, 8, 8)
    assert batch["move_index"].shape == (5,)
    assert batch["value"].shape == (5,)


def test_replay_buffer_eviction():
    buf = ReplayBuffer(capacity=4)
    for i in range(10):
        buf.add(_sample(version=i))
    assert len(buf) == 4
    versions = set(buf.to_numpy()["version"].tolist())
    assert versions == {6, 7, 8, 9}


def test_replay_buffer_save_load_round_trip(tmp_path):
    buf = ReplayBuffer(capacity=100)
    buf.extend([_sample(version=i % 3) for i in range(20)])
    path = tmp_path / "buf.npz"
    buf.save(str(path))
    other = ReplayBuffer(capacity=100)
    other.load(str(path))
    assert len(other) == 20
    np.testing.assert_array_equal(other.to_numpy()["input"], buf.to_numpy()["input"])
    np.testing.assert_array_equal(other.to_numpy()["value"], buf.to_numpy()["value"])


def test_replay_buffer_empty_sample_raises():
    buf = ReplayBuffer(capacity=10)
    with pytest.raises(ValueError):
        buf.sample(4)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
def test_dataset_chunk_write_read_round_trip(tmp_path):
    samples = [_sample(version=i) for i in range(5)]
    path = str(tmp_path / "chunk_000000.npz")
    write_chunk(path, samples)
    loaded = read_chunk(path)
    assert len(loaded) == 5
    for a, b in zip(samples, loaded):
        np.testing.assert_array_equal(a["input"], b["input"])
        assert a["move_index"] == b["move_index"]
        assert a["value"] == b["value"]


def test_chunk_writer_flushes_and_dataset_reads(tmp_path):
    writer = ChunkWriter(str(tmp_path), chunk_size=3)
    writer.add([_sample(version=i) for i in range(7)])
    writer.flush()
    chunks = list_chunks(str(tmp_path))
    assert len(chunks) == 3  # 3 + 3 + 1
    ds = SelfPlayDataset(chunks)
    assert len(ds) == 7
    item = ds[0]
    assert item["input"].shape == (NUM_CHANNELS, 8, 8)
    assert isinstance(item["value"], torch.Tensor)


# ----------------------------------------------------------------------
# Trainer / Learner: learning must actually change weights
# ----------------------------------------------------------------------
def test_trainer_changes_weights():
    torch.manual_seed(0)
    model = create_chess_net(channels=8, res_blocks=0)
    buf = ReplayBuffer(capacity=200)
    samples = [_sample() for _ in range(48)]
    buf.extend(samples)
    trainer = Trainer(model=model, device="cpu", batch_size=16, learning_rate=0.1)
    before = trainer.parameter_norm()
    trainer.train(buf, steps=3)
    after = trainer.parameter_norm()
    assert abs(after - before) > 1e-6, "weights must change after training"


def test_trainer_policy_and_value_losses_decrease():
    torch.manual_seed(1)
    model = create_chess_net(channels=8, res_blocks=0)
    buf = ReplayBuffer(capacity=256)
    buf.extend([_sample() for _ in range(128)])
    trainer = Trainer(model=model, device="cpu", batch_size=32, learning_rate=0.05)
    first = trainer.train(buf, steps=1)
    # A few more steps should reduce the total loss (same fixed data).
    last = trainer.train(buf, steps=6)
    assert last["total_loss"] < first["total_loss"]


# ----------------------------------------------------------------------
# Model manager / checkpointing / resume
# ----------------------------------------------------------------------
def test_checkpoint_and_resume_round_trip(tmp_path):
    torch.manual_seed(2)
    model = create_chess_net(channels=16, res_blocks=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mm = ModelManager(str(tmp_path))
    meta = CheckpointMetadata(version=7, games_trained=123, training_steps=456,
                              timestamp="now", config={"a": 1})
    path = mm.save(model, optimizer, meta)
    assert mm.find_latest() == path
    assert mm.next_version() == 8

    model2 = create_chess_net(channels=16, res_blocks=1)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.1)
    mm2 = ModelManager(str(tmp_path))
    restored = mm2.resume(model2, optimizer2)
    assert restored is not None
    assert restored.version == 7
    assert restored.games_trained == 123
    # Weights match.
    s1 = model.state_dict()
    s2 = model2.state_dict()
    for k in s1:
        torch.testing.assert_close(s1[k], s2[k])


def test_checkpoint_versioning(tmp_path):
    mm = ModelManager(str(tmp_path))
    assert mm.next_version() == 1
    model = create_chess_net(channels=8, res_blocks=0)
    opt = torch.optim.Adam(model.parameters())
    mm.save(model, opt, CheckpointMetadata(version=1, games_trained=0, training_steps=0, timestamp="t"))
    mm.save(model, opt, CheckpointMetadata(version=2, games_trained=5, training_steps=5, timestamp="t"))
    versions = [v for v, _ in mm.list_checkpoints()]
    assert versions == [1, 2]


def test_learner_version_and_weights_change_after_train(tmp_path):
    torch.manual_seed(3)
    model = create_chess_net(channels=16, res_blocks=1)
    buf = ReplayBuffer(capacity=256)
    buf.extend([_sample() for _ in range(128)])
    learner = Learner(model=model, replay_buffer=buf, model_manager=ModelManager(str(tmp_path)),
                      device="cpu", batch_size=16, training_steps=2, learning_rate=0.05)
    before_norm = learner.trainer.parameter_norm()
    summary = learner.train()
    assert summary["param_delta"] > 0
    assert summary["training_steps_total"] == 2
    # checkpoint bumps via explicit version passed by the coordinator
    learner.checkpoint(games_trained=10, version=5)
    assert learner.model_manager.find_latest() is not None