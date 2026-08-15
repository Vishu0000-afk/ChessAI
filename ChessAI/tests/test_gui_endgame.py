"""Tests for the GUI game-over reason helper."""

import chess

from src.gui.chess_gui import game_over_reason


def test_checkmate_reason():
    board = chess.Board()
    board.push_san("f3")
    board.push_san("e5")
    board.push_san("g4")
    board.push_san("Qh4#")
    assert board.is_checkmate()
    assert game_over_reason(board) == "Checkmate — Black wins!"


def test_stalemate_reason():
    board = chess.Board("7k/8/6Q1/5K2/8/8/8/8 b - - 0 1")
    assert board.is_stalemate()
    assert game_over_reason(board) == "Stalemate — draw"


def test_insufficient_material_reason():
    board = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1")
    assert board.is_insufficient_material()
    assert game_over_reason(board) == "Insufficient material — draw"


def test_fivefold_repetition_reason():
    board = chess.Board()
    cycle = ["b1c3", "g8f6", "g1f3", "b8c6", "c3b1", "f6g8", "f3g1", "c6b8"]
    for i in range(32):
        board.push(chess.Move.from_uci(cycle[i % 8]))
    assert board.is_fivefold_repetition()
    assert not board.is_checkmate()
    assert not board.is_stalemate()
    assert game_over_reason(board) == "Draw — repetition (fivefold)"


def test_seventyfive_move_rule_reason():
    board = chess.Board("8/8/8/4k3/8/8/4K3/4Q3 w - - 150 60")
    assert board.is_seventyfive_moves()
    assert not board.is_insufficient_material()
    assert game_over_reason(board) == "Draw — 75-move rule"


def test_game_in_progress():
    assert game_over_reason(chess.Board()) == "Game in progress"