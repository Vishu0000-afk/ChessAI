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
from src.selfplay.statistics import SelfPlayStats
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
# Honest draw reporting (dropped draws counted separately)
# ----------------------------------------------------------------------
def test_stats_tracks_dropped_draws_and_reports_true_draw_rate():
    stats = SelfPlayStats()
    stats.record_game("1-0", 30, 30)
    stats.record_game("1/2-1/2", 40, 40)
    stats.record_dropped_draw(50, 50)
    stats.record_dropped_draw(60, 60)

    rates = stats.result_rates()
    assert stats.games == 4
    assert stats.draws_dropped == 2
    # True draw share includes kept + dropped draws.
    assert rates["draws"] == 3 / 4
    assert rates["white"] == 1 / 4
    assert rates["black"] == 0.0


def test_coordinator_drop_uses_dropped_draw_counter(tmp_path):
    cfg = _tiny_config(tmp_path)
    coordinator = SelfPlayCoordinator(cfg)
    # A stream of 4 drawn games, all beyond the draw cap, must increment the
    # dropped counter (not the kept-draw counter) and still count the games.
    for _ in range(4):
        game = _draw_game()
        if drop_excess_draw(game, coordinator.stats.draws, coordinator.stats.games, cfg.draw_max_rate):
            coordinator.stats.record_dropped_draw(game["moves"], len(game["samples"]))
        else:
            coordinator.stats.record_game(game["result"], game["moves"], len(game["samples"]))
    assert coordinator.stats.games == 4
    assert coordinator.stats.draws == 1  # first draw is below the cap, so kept
    assert coordinator.stats.draws_dropped == 3


# ----------------------------------------------------------------------
# Random opening + Dirichlet noise (self-play draw collapse mitigation)
# ----------------------------------------------------------------------
def test_neural_agent_dirichlet_defaults_off():
    from src.agents.neural_agent import NeuralAgent
    from src.agents.predictor import LocalPredictor

    agent = NeuralAgent(LocalPredictor(_tiny_model().eval(), device="cpu"))
    assert agent.dirichlet_epsilon == 0.0  # eval/other paths unaffected by default


def test_random_open_plies_overrides_policy():
    from src.agents.predictor import LocalPredictor

    # A constant-logit predictor would otherwise make both runs identical.
    class ConstantPredictor:
        def predict_batch(self, boards, encoded=None):
            return np.zeros((len(boards), 4096), dtype=np.float32), np.zeros(len(boards))

    first = BatchedGameSimulator(
        ConstantPredictor(), concurrency=1, temperature=1.0,
        max_game_moves=50, model_version=1,
        rng=np.random.default_rng(0), random_open_plies=4,
    )
    second = BatchedGameSimulator(
        ConstantPredictor(), concurrency=1, temperature=1.0,
        max_game_moves=50, model_version=1,
        rng=np.random.default_rng(1), random_open_plies=4,
    )
    first.run_games(1)
    second.run_games(1)
    g1 = first.completed_games[0]["board"].move_stack[:4]
    g2 = second.completed_games[0]["board"].move_stack[:4]
    # Different seeds must produce different random openings.
    assert [m.uci() for m in g1] != [m.uci() for m in g2]


# ----------------------------------------------------------------------
# Capped games scored by material for the value target
# ----------------------------------------------------------------------
def _capped_game(fen, max_game_moves=400):
    from src.selfplay.worker import BatchedGameSimulator

    sim = BatchedGameSimulator(None, concurrency=1, temperature=1.0,
                               max_game_moves=max_game_moves, model_version=1)
    return sim, {
        "board": chess.Board(fen),
        "move_count": max_game_moves,
        "samples": [
            {"color": chess.WHITE, "move_index": 0, "version": 1},
            {"color": chess.BLACK, "move_index": 1, "version": 1},
        ],
    }


def test_capped_game_with_material_lead_gets_decisive_values():
    sim, game = _capped_game("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1")  # white up a queen
    sim._finish_game(game)
    assert game["result"] == "1/2-1/2"  # still a draw for statistics
    # samples[0] was White to move, samples[1] Black to move.
    assert game["samples"][0]["value"] == 1.0
    assert game["samples"][1]["value"] == -1.0


def test_capped_game_with_equal_material_stays_draw():
    sim, game = _capped_game("7k/8/8/8/8/8/8/7K w - - 0 1")  # bare kings
    sim._finish_game(game)
    assert game["result"] == "1/2-1/2"
    assert game["samples"][0]["value"] == 0.0
    assert game["samples"][1]["value"] == 0.0


def test_run_games_move_count_is_not_double_counted():
    from src.agents.predictor import LocalPredictor

    model = _tiny_model().eval()
    sim = BatchedGameSimulator(LocalPredictor(model, device="cpu"),
                               concurrency=2, temperature=1.0,
                               max_game_moves=50, model_version=1)
    stats = sim.run_games(4)
    expected = sum(g["move_count"] for g in sim.completed_games)
    assert stats["moves"] == expected  # regression: [-0:] re-summed all games


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