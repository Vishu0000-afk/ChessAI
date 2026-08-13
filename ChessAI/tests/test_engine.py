"""Tests for the high-level ChessEngine class."""

import chess

from src.engine.board import Board
from src.engine.engine import ChessEngine


def test_engine_initializes_with_defaults():
    engine = ChessEngine()
    assert engine.depth == 3
    assert engine.transposition_table.enabled is True


def test_engine_can_search_and_return_legal_move():
    engine = ChessEngine(depth=2)
    board = Board.new_game()
    move = engine.get_best_move(board)
    assert move is not None
    assert move in board.legal_moves()


def test_engine_stats_populated_after_search():
    engine = ChessEngine(depth=2)
    board = Board.new_game()
    engine.get_best_move(board)
    stats = engine.get_stats()
    assert stats.depth == 2
    assert stats.nodes_searched > 0
    assert stats.search_time_seconds >= 0


def test_engine_returns_none_when_game_over():
    engine = ChessEngine(depth=2)
    board = Board.from_fen("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    board.make_move_uci("e1e8")
    assert board.is_checkmate()
    move = engine.get_best_move(board)
    assert move is None


def test_engine_reset_clears_transposition_table():
    engine = ChessEngine(depth=2)
    board = Board.new_game()
    engine.get_best_move(board)
    engine.reset()
    assert len(engine.transposition_table) == 0


def test_engine_with_transposition_table_disabled():
    engine = ChessEngine(depth=2, use_transposition_table=False)
    board = Board.new_game()
    move = engine.get_best_move(board)
    assert move in board.legal_moves()


def test_evaluate_position_returns_int():
    engine = ChessEngine(depth=2)
    board = Board.new_game()
    score = engine.evaluate_position(board)
    assert isinstance(score, int)
