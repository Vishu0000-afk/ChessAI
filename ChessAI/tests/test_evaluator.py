"""Tests for the classical Evaluator."""

import chess

from src.engine.evaluator import Evaluator


def test_starting_position_is_approximately_equal():
    evaluator = Evaluator()
    board = chess.Board()
    score = evaluator.evaluate(board)
    assert abs(score) <= 50  # Small PST/mobility asymmetry is fine; material is 0.


def test_material_advantage_direction_white_up_a_queen():
    evaluator = Evaluator()
    # White has an extra queen compared to the starting position.
    board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    score = evaluator.evaluate(board)
    assert score > 0


def test_material_advantage_direction_black_up_a_rook():
    evaluator = Evaluator()
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1")
    score = evaluator.evaluate(board)
    assert score < 0


def test_capturing_a_queen_produces_large_evaluation_swing():
    evaluator = Evaluator()
    # Position where White can capture Black's queen with a rook.
    board = chess.Board("3q4/8/8/8/8/8/8/3RK2k w - - 0 1")
    score_before = evaluator.evaluate(board)

    board.push(chess.Move.from_uci("d1d8"))
    score_after = evaluator.evaluate(board)

    swing = score_after - score_before
    assert swing > 800  # Roughly a queen's value.


def test_checkmate_scores_as_extreme_for_mated_side():
    evaluator = Evaluator()
    # Fool's mate: Black delivers mate, White (to move) is mated.
    board = chess.Board()
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        board.push(chess.Move.from_uci(uci))
    assert board.is_checkmate()
    score = evaluator.evaluate(board)
    assert score < -100_000  # White (to move) is mated -> very negative.


def test_stalemate_scores_zero():
    evaluator = Evaluator()
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert board.is_stalemate()
    assert evaluator.evaluate(board) == 0
