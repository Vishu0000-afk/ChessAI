"""Tests for the alpha-beta Search engine."""

import chess

from src.engine.evaluator import Evaluator
from src.engine.search import Search
from src.engine.transposition import TranspositionTable


def make_search(use_tt: bool = True, use_move_ordering: bool = True) -> Search:
    evaluator = Evaluator()
    tt = TranspositionTable(enabled=use_tt)
    return Search(evaluator=evaluator, transposition_table=tt, use_move_ordering=use_move_ordering)


def test_returns_legal_move_from_starting_position():
    search = make_search()
    board = chess.Board()
    result = search.search(board, depth=2)
    assert result.best_move is not None
    assert result.best_move in board.legal_moves


def test_finds_obvious_capture():
    search = make_search()
    # White rook can capture a hanging queen.
    board = chess.Board("3q4/8/8/8/8/8/8/3RK2k w - - 0 1")
    result = search.search(board, depth=2)
    assert result.best_move == chess.Move.from_uci("d1d8")


def test_recognizes_checkmate_in_one():
    search = make_search()
    # White to move: Qh5-h7 style back-rank style mate setup.
    # Simple forced mate in 1: Black king boxed in, White queen delivers mate.
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    # Rook can deliver mate on e8.
    result = search.search(board, depth=2)
    board.push(result.best_move)
    assert board.is_checkmate()


def test_avoids_illegal_moves_across_many_searches():
    search = make_search()
    board = chess.Board()
    moves_played = 0
    while not board.is_game_over() and moves_played < 6:
        result = search.search(board, depth=2)
        assert result.best_move in board.legal_moves
        board.push(result.best_move)
        moves_played += 1


def test_handles_check_position_correctly():
    search = make_search()
    # White king in check from Black queen; White must respond legally.
    board = chess.Board("4k3/8/8/8/8/8/8/4K2q w - - 0 1")
    assert board.is_check()
    result = search.search(board, depth=2)
    assert result.best_move in board.legal_moves
    board.push(result.best_move)
    assert not board.is_check() or board.turn == chess.BLACK


def test_search_with_transposition_table_disabled_still_works():
    search = make_search(use_tt=False)
    board = chess.Board()
    result = search.search(board, depth=2)
    assert result.best_move in board.legal_moves


def test_node_count_is_positive():
    search = make_search()
    board = chess.Board()
    result = search.search(board, depth=2)
    assert result.nodes_searched > 0


def test_no_legal_moves_returns_none_move():
    search = make_search()
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    board.push(chess.Move.from_uci("e1e8"))  # White delivers checkmate.
    assert board.is_checkmate()
    result = search.search(board, depth=2)
    assert result.best_move is None
