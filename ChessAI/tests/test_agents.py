"""Tests for the agent implementations."""

import chess
import numpy as np

from src.agents.classical_engine import ClassicalEngineAgent
from src.agents.neural_agent import NeuralAgent
from src.agents.predictor import LocalPredictor
from src.agents.random_agent import RandomAgent
from src.learning.network import create_chess_net


def test_random_agent_always_plays_legal_moves():
    agent = RandomAgent()
    board = chess.Board()
    for _ in range(20):
        move = agent.select_move(board)
        assert move in board.legal_moves
        board.push(move)
        if board.is_game_over():
            break


def test_random_agent_returns_none_when_game_over():
    board = chess.Board()
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:  # Fool's mate
        board.push(chess.Move.from_uci(uci))
    assert board.is_checkmate()
    agent = RandomAgent()
    assert agent.select_move(board) is None


def test_classical_engine_agent_returns_legal_move():
    agent = ClassicalEngineAgent(depth=2)
    board = chess.Board()
    move = agent.select_move(board)
    assert move in board.legal_moves


def _tiny_net():
    return create_chess_net(channels=16, res_blocks=1)


def test_neural_agent_returns_legal_move():
    model = _tiny_net().eval()
    predictor = LocalPredictor(model, device="cpu")
    agent = NeuralAgent(predictor, temperature=0.0)
    board = chess.Board()
    for _ in range(12):
        move = agent.select_move(board)
        assert move in board.legal_moves
        board.push(move)
        if board.is_game_over():
            break


def test_neural_agent_never_selects_illegal_move_when_sampling():
    model = _tiny_net().eval()
    predictor = LocalPredictor(model, device="cpu")
    agent = NeuralAgent(predictor, temperature=2.0)
    board = chess.Board()
    for _ in range(20):
        move = agent.select_move(board)
        assert move in board.legal_moves
        board.push(move)


def test_neural_agent_batched_equals_single():
    model = _tiny_net().eval()
    predictor = LocalPredictor(model, device="cpu")
    boards = [chess.Board(), chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")]
    single = [predictor.predict_batch([b]) for b in boards]
    batched = predictor.predict_batch(boards)
    for i in range(len(boards)):
        np.testing.assert_allclose(single[i][0][0], batched[0][i], atol=1e-6)
        np.testing.assert_allclose(single[i][1][0], batched[1][i], atol=1e-6)


def test_neural_agent_batch_result_shapes():
    model = _tiny_net().eval()
    predictor = LocalPredictor(model, device="cpu")
    agent = NeuralAgent(predictor)
    boards = [chess.Board(), chess.Board("8/8/8/4k3/8/8/8/4K3 w - - 0 1")]
    logits, values = agent.predict_batch(boards)
    assert logits.shape == (2, 4096)
    assert values.shape == (2,)