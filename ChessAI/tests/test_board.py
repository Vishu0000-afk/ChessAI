"""Tests for the Board abstraction."""

import chess
import pytest

from src.engine.board import Board


def test_new_game_starting_position():
    board = Board.new_game()
    assert board.get_fen().startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq")
    assert board.turn() == chess.WHITE
    assert len(board.legal_moves()) == 20


def test_legal_move_generation_count_after_e4():
    board = Board.new_game()
    board.make_move_uci("e2e4")
    assert board.turn() == chess.BLACK
    assert len(board.legal_moves()) == 20


def test_making_moves_updates_history():
    board = Board.new_game()
    board.make_move_uci("e2e4")
    board.make_move_uci("e7e5")
    assert board.get_move_history() == ["e2e4", "e7e5"]


def test_undo_move_restores_position():
    board = Board.new_game()
    fen_before = board.get_fen()
    board.make_move_uci("e2e4")
    undone = board.undo_move()
    assert undone == chess.Move.from_uci("e2e4")
    assert board.get_fen() == fen_before


def test_undo_with_no_moves_returns_none():
    board = Board.new_game()
    assert board.undo_move() is None


def test_illegal_move_raises():
    board = Board.new_game()
    with pytest.raises(ValueError):
        board.make_move_uci("e2e5")


def test_fen_loading():
    fen = "8/8/8/4k3/8/8/8/4K2R w K - 0 1"
    board = Board.from_fen(fen)
    assert board.get_fen() == fen


def test_check_detection():
    # Fool's mate position: Black delivers checkmate on move 2.
    board = Board.new_game()
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        board.make_move_uci(uci)
    assert board.is_checkmate()
    assert board.is_check()


def test_stalemate_detection():
    # Known stalemate position.
    board = Board.from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert board.is_stalemate()
    assert not board.is_checkmate()


def test_copy_is_independent():
    board = Board.new_game()
    board_copy = board.copy()
    board_copy.make_move_uci("e2e4")
    assert board.get_move_history() == []
    assert board_copy.get_move_history() == ["e2e4"]
