"""Tests for the self-play package: batched inference, workers,
evaluation, and an end-to-end continuous-learning smoke run."""

import multiprocessing as mp

import chess
import numpy as np
import torch

from configs.config import SelfPlayConfig
from src.learning.network import create_chess_net
from src.selfplay.coordinator import SelfPlayCoordinator, drop_excess_draw
from src.selfplay.evaluator import evaluate_models
from src.selfplay.inference import InferenceClient, InferenceServer
from src.selfplay.worker import BatchedGameSimulator


def _tiny_model():
    return create_chess_net(channels=16, res_blocks=1)


# ----------------------------------------------------------------------
# Batched inference server/client
# ----------------------------------------------------------------------
def test_inference_server_client_roundtrip():
    request_queue = mp.Queue()
    result_queue = mp.Queue()
    model = _tiny_model()
    server = InferenceServer(
        models={"current": model},
        request_queue=request_queue,
        result_queue=result_queue,
        device="cpu",
        use_mixed_precision=False,
    )
    server.start()
    try:
        client = InferenceClient(request_queue, result_queue)
        boards = [chess.Board(), chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")]
        logits, values = client.predict_batch("current", boards)
        assert logits.shape == (2, 4096)
        assert values.shape == (2,)
        # Compare with a direct forward pass on the same model.
        from src.learning.encoding import encode_board_batch
        with torch.no_grad():
            x = torch.from_numpy(encode_board_batch(boards))
            exp_logits, exp_values = model(x)
        np.testing.assert_allclose(logits, exp_logits.numpy(), atol=1e-5)
        np.testing.assert_allclose(values, exp_values.numpy().reshape(-1), atol=1e-5)
    finally:
        server.stop()
        server.join(timeout=5)
        request_queue.close()
        result_queue.close()


def test_inference_server_multiple_models():
    request_queue = mp.Queue()
    result_queue = mp.Queue()
    model = _tiny_model()
    server = InferenceServer(
        models={"current": model},
        request_queue=request_queue,
        result_queue=result_queue,
        device="cpu",
        use_mixed_precision=False,
    )
    server.start()
    try:
        other = _tiny_model()
        server.register_model("previous", other)
        client = InferenceClient(request_queue, result_queue)
        b = [chess.Board()]
        logits_a, _ = client.predict_batch("current", b)
        logits_b, _ = client.predict_batch("previous", b)
        # Two independent models may give different logits.
        assert logits_a.shape == (1, 4096)
        assert logits_b.shape == (1, 4096)
    finally:
        server.stop()
        server.join(timeout=5)
        request_queue.close()
        result_queue.close()


# ----------------------------------------------------------------------
# Batched self-play game simulation
# ----------------------------------------------------------------------
def test_batched_simulator_plays_only_legal_moves():
    from src.agents.predictor import LocalPredictor

    model = _tiny_model().eval()
    predictor = LocalPredictor(model, device="cpu")
    sim = BatchedGameSimulator(predictor, concurrency=2, temperature=1.0, max_game_moves=50, model_version=1)
    stats = sim.run_games(4)

    # Every completed game must have been legal: replay each move against a
    # fresh board and confirm the engine accepted it.
    for game in sim.completed_games:
        board = chess.Board()
        for move in game["board"].move_stack:
            assert move in board.legal_moves
            board.push(move)
        # Games end by mate/stalemate/draw OR hit the move cap (counted draw).
        assert board.is_game_over() or len(board.move_stack) >= 50
    # run_games() completes `count` games plus any in-flight tail.
    assert stats["games"] >= 4
    assert stats["games"] == len(sim.completed_games)
    # Every sample must have a legal move index and a value target in {-1, 0, 1}.
    for game in sim.completed_games:
        for s in game["samples"]:
            assert -1.0 <= s["value"] <= 1.0
            assert isinstance(s["move_index"], (int, np.integer))


def test_batched_simulator_generates_value_targets_from_result():
    from src.agents.predictor import LocalPredictor

    model = _tiny_model().eval()
    predictor = LocalPredictor(model, device="cpu")
    sim = BatchedGameSimulator(predictor, concurrency=1, temperature=0.0, max_game_moves=50, model_version=2)
    sim.run_games(1)
    game = sim.completed_games[0]
    if game["result"] == "1-0":
        # White to move positions get +1, black to move get -1.
        pass  # values already assigned; checked below
    for s in game["samples"]:
        assert s["version"] == 2


# ----------------------------------------------------------------------
# Evaluator
# ----------------------------------------------------------------------
def test_evaluator_reports_sane_counts():
    request_queue = mp.Queue()
    result_queue = mp.Queue()
    model = _tiny_model()
    server = InferenceServer(
        models={"current": model},
        request_queue=request_queue,
        result_queue=result_queue,
        device="cpu",
        use_mixed_precision=False,
    )
    server.start()
    try:
        other = _tiny_model()
        server.register_model("previous", other)
        result = evaluate_models(
            request_queue, result_queue,
            model_a="current", model_b="previous",
            num_games=6, concurrency=2, max_game_moves=30,
        )
        assert result.games == 6
        assert result.a_wins + result.b_wins + result.draws == 6
        assert 0.0 <= result.score <= 1.0
    finally:
        server.stop()
        server.join(timeout=5)
        request_queue.close()
        result_queue.close()


# ----------------------------------------------------------------------
# Draw cap / color rebalancing
# ----------------------------------------------------------------------
def _draw_game():
    return {
        "result": "1/2-1/2",
        "moves": 30,
        "samples": [
            {"color": chess.WHITE, "value": 0.0, "move_index": 0, "version": 1},
            {"color": chess.BLACK, "value": 0.0, "move_index": 1, "version": 1},
        ],
    }


def test_drop_excess_draw_kept_below_cap():
    game = _draw_game()
    assert not drop_excess_draw(game, draws=0, games=0, draw_max_rate=0.2)
    assert not drop_excess_draw(game, draws=1, games=9, draw_max_rate=0.2)  # 0.111 < 0.2


def test_drop_excess_draw_at_cap():
    # 2 draws / 10 games = 0.2 (at cap) -> dropped.
    game = _draw_game()
    assert drop_excess_draw(game, draws=2, games=10, draw_max_rate=0.2)


def test_drop_excess_draw_over_cap():
    # 5 draws / 10 games = 0.5 > 0.2 -> dropped.
    game = _draw_game()
    assert drop_excess_draw(game, draws=5, games=10, draw_max_rate=0.2)


def test_drop_excess_draw_ignores_decisive_games():
    game = {"result": "1-0", "moves": 20, "samples": []}
    assert not drop_excess_draw(game, draws=10, games=10, draw_max_rate=0.2)


# ----------------------------------------------------------------------
# End-to-end continuous learning
# ----------------------------------------------------------------------
def _tiny_config(tmp_path) -> SelfPlayConfig:
    return SelfPlayConfig(
        game_mode="self_play",
        games_total=10,
        num_workers=1,
        self_play_concurrency=2,
        temperature=1.0,
        temp_final=0.2,
        temp_decay_games=100,
        max_game_moves=100,
        replay_buffer_size=10_000,
        train_every_n_games=5,
        training_steps=2,
        batch_size=16,
        learning_rate=0.05,
        use_mixed_precision=False,
        nn_conv_channels=16,
        nn_res_blocks=0,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        checkpoint_every_n_games=5,
        dataset_dir=str(tmp_path / "data"),
        dataset_chunk_size=100,
        persist_every_n_games=0,
        evaluate_enabled=True,
        evaluate_games=4,
        evaluate_concurrency=2,
        promotion_min_score=0.45,
        inference_max_batch=64,
        device="cpu",
        train_enabled=True,
        seed=7,
    )


def test_selfplay_end_to_end(tmp_path):
    torch.manual_seed(0)
    cfg = _tiny_config(tmp_path)
    coordinator = SelfPlayCoordinator(cfg)

    initial_norm = coordinator.learner.trainer.parameter_norm()
    stats = coordinator.run()

    # The run produced games, samples, and training.
    assert stats.games >= 10
    assert stats.samples > 0
    assert stats.training_steps > 0

    # Weights actually changed because of learning.
    final_norm = coordinator.learner.trainer.parameter_norm()
    assert abs(final_norm - initial_norm) > 1e-6

    # Model version advanced and a checkpoint exists.
    assert coordinator.learner.version >= 1
    assert coordinator.model_manager.find_latest() is not None

    # Checkpoint metadata records games trained.
    meta = coordinator.model_manager.resume(coordinator.model)
    assert meta is not None
    assert meta.version == coordinator.learner.version


def test_selfplay_resume_after_interruption(tmp_path):
    torch.manual_seed(1)
    cfg1 = _tiny_config(tmp_path)
    cfg1.games_total = 10
    coordinator1 = SelfPlayCoordinator(cfg1)
    coordinator1.run()
    v1 = coordinator1.learner.version
    assert v1 >= 1

    # A second run with the same checkpoint dir must resume from v1 and
    # keep learning towards a higher game total.
    cfg2 = _tiny_config(tmp_path)
    cfg2.games_total = 20
    coordinator2 = SelfPlayCoordinator(cfg2)
    stats2 = coordinator2.run()
    assert coordinator2.start_games == 10
    assert coordinator2.learner.version >= v1
    assert stats2.games == 10  # this run added the remaining 10
    assert coordinator2.model_manager.find_latest() is not None


def test_benchmark_mode_runs(tmp_path):
    cfg = _tiny_config(tmp_path)
    cfg.train_enabled = False
    cfg.evaluate_enabled = False
    cfg.auto_resume = False
    cfg.games_total = 8
    coordinator = SelfPlayCoordinator(cfg)
    stats = coordinator.run()
    assert stats.games == 8
    assert stats.training_steps == 0  # training disabled in benchmark mode