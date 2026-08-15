"""Parallel self-play workers and batched game simulation.

Workers are separate processes. They never own a neural model: every batch of
positions is sent through the shared inference server (see ``inference.py``).
A worker keeps ``concurrency`` games in flight and plays them in lockstep, so
each inference call already contains ``concurrency`` positions; many workers
multiply this into large GPU batches.

Experiences (position, policy target, legal mask, value target, model version)
are emitted per finished game to the coordinator's experience queue.
"""

from __future__ import annotations

import logging
import random
import sys
import time
from multiprocessing import Process
from typing import Dict, List, Optional

import chess
import numpy as np

from src.agents.neural_agent import NeuralAgent
from src.learning.encoding import (
    encode_board_batch,
    legal_move_mask,
    move_to_index,
    pack_mask,
)
from src.selfplay.inference import DEFAULT_MODEL_ID, InferenceClient

logger = logging.getLogger(__name__)

# Material values used to score games that hit the move cap (a real, if crude,
# proxy for who is actually winning when shuffling ends in a forced draw).
_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def _material_balance(board: chess.Board) -> float:
    """White material minus black material, in pawn units."""
    balance = 0.0
    for square, piece in board.piece_map().items():
        sign = 1.0 if piece.color == chess.WHITE else -1.0
        balance += sign * _PIECE_VALUES.get(piece.piece_type, 0)
    return balance


# =============================================================================
# Batched game simulation core (shared by self-play and evaluation)
# =============================================================================
class BatchedGameSimulator:
    """Plays `concurrency` games in lockstep, batched through one predictor."""

    def __init__(
        self,
        predictor,
        concurrency: int,
        temperature: float,
        max_game_moves: int,
        model_version: Optional[int],
        rng: Optional[np.random.Generator] = None,
        random_open_plies: int = 0,
        dirichlet_epsilon: float = 0.0,
        dirichlet_alpha: float = 0.03,
    ) -> None:
        self.predictor = predictor
        self.concurrency = concurrency
        self.max_game_moves = max_game_moves
        self.model_version = model_version
        self.rng = rng or np.random.default_rng()
        self.random_open_plies = random_open_plies

        self._active: List[Optional[Dict]] = []
        self.completed_games: List[Dict] = []
        self.completed_samples: List[List[Dict]] = []
        self.agents = [
            NeuralAgent(
                predictor,
                temperature=temperature,
                version=model_version,
                dirichlet_epsilon=dirichlet_epsilon,
                dirichlet_alpha=dirichlet_alpha,
            ),
            NeuralAgent(
                predictor,
                temperature=temperature,
                version=model_version,
                dirichlet_epsilon=dirichlet_epsilon,
                dirichlet_alpha=dirichlet_alpha,
            ),
        ]

    # ------------------------------------------------------------------
    def _new_game(self) -> Dict:
        return {
            "board": chess.Board(),
            "agent_index": len(self._active) % 2,  # alternate colors per game
            "move_count": 0,
            "samples": [],
            "agent": self.agents[len(self._active) % 2],
        }

    def _refill(self) -> None:
        while len(self._active) < self.concurrency:
            self._active.append(self._new_game())

    def step(self) -> int:
        """Advance all active games one ply; returns newly completed games."""
        if not self._active:
            return 0

        boards = [g["board"] for g in self._active]
        encoded = encode_board_batch(boards)
        logits, _ = self.predictor.predict_batch(boards, encoded=encoded)

        completed = 0
        for i, game in enumerate(self._active):
            board = game["board"]
            moves = list(board.legal_moves)
            move = None
            if moves and not board.is_game_over() and game["move_count"] < self.max_game_moves:
                if game["move_count"] < self.random_open_plies:
                    move = self.rng.choice(moves)
                else:
                    move = game["agent"]._select_from_logits(board, moves, logits[i])

            if move is None:
                self._finish_game(game)
                self._active[i] = None
                completed += 1
                continue

            sample = {
                "input": encoded[i],
                "move_index": move_to_index(move, board.turn),
                "legal_packed": pack_mask(legal_move_mask(board)),
                "color": board.turn,
                "version": game["agent"].version or 0,
                "value": 0.0,
            }
            game["samples"].append(sample)
            board.push(move)
            game["move_count"] += 1

            if board.is_game_over() or game["move_count"] >= self.max_game_moves:
                self._finish_game(game)
                self._active[i] = None
                completed += 1

        self._active = [g for g in self._active if g is not None]
        return completed

    def _finish_game(self, game: Dict) -> None:
        board = game["board"]
        if board.is_game_over():
            result = board.result(claim_draw=True)
            white_won = result == "1-0"
            black_won = result == "0-1"
        else:
            # Capped by max_game_moves: keep "1/2-1/2" for the statistics, but
            # score the training value by material so long shuffling games still
            # carry a real, decisive signal instead of always predicting 0.
            result = "1/2-1/2"
            balance = _material_balance(board)
            white_won = balance >= 1.0
            black_won = balance <= -1.0

        for s in game["samples"]:
            if s["color"] == chess.WHITE:
                s["value"] = 1.0 if white_won else (-1.0 if black_won else 0.0)
            else:
                s["value"] = 1.0 if black_won else (-1.0 if white_won else 0.0)
            del s["color"]

        game["result"] = result
        self.completed_games.append(game)

    def run_games(self, count: int) -> Dict:
        """Simulate `count` games plus the in-flight tail; returns stats."""
        completed = 0
        total_moves = 0
        while completed < count:
            self._refill()
            new = self.step()
            start = len(self.completed_games) - new
            for g in self.completed_games[start:]:
                total_moves += g["move_count"]
            completed += new
        # Finish any games still in flight (no new games are started).
        while self._active:
            new = self.step()
            start = len(self.completed_games) - new
            for g in self.completed_games[start:]:
                total_moves += g["move_count"]
            completed += new
        return {"games": completed, "moves": total_moves}

    def pop_completed(self) -> List[Dict]:
        """Return and clear all completed games (for streaming experiences)."""
        games = self.completed_games
        self.completed_games = []
        return games


# =============================================================================
# Self-play worker process
# =============================================================================
def run_selfplay_worker(
    request_queue,
    result_queue,
    experience_queue,
    games_target: int,
    global_start: int,
    games_global: int,
    concurrency: int,
    temperature: float,
    temp_final: float,
    temp_decay_games: int,
    max_game_moves: int,
    model_version: int,
    seed: int,
    stop_event,
    flush_every: int = 5,
    worker_id: int = 0,
    random_open_plies: int = 0,
    dirichlet_epsilon: float = 0.0,
    dirichlet_alpha: float = 0.03,
) -> None:
    """Top-level worker function (picklable: works with fork and spawn)."""
    try:
        _worker_loop(
            request_queue,
            result_queue,
            experience_queue,
            games_target,
            global_start,
            games_global,
            concurrency,
            temperature,
            temp_final,
            temp_decay_games,
            max_game_moves,
            model_version,
            seed,
            stop_event,
            flush_every,
            worker_id,
            random_open_plies,
            dirichlet_epsilon,
            dirichlet_alpha,
        )
    except Exception:
        logger.exception("Self-play worker crashed")
        sys.exit(1)


def _worker_loop(
    request_queue,
    result_queue,
    experience_queue,
    games_target: int,
    global_start: int,
    games_global: int,
    concurrency: int,
    temperature: float,
    temp_final: float,
    temp_decay_games: int,
    max_game_moves: int,
    model_version: int,
    seed: int,
    stop_event,
    flush_every: int,
    worker_id: int = 0,
    random_open_plies: int = 0,
    dirichlet_epsilon: float = 0.0,
    dirichlet_alpha: float = 0.03,
) -> None:
    client = InferenceClient(request_queue, result_queue, worker_id=worker_id)

    class _QueuePredictor:
        def __init__(self, client, model_id: str):
            self.client = client
            self.model_id = model_id

        def predict_batch(self, boards, encoded=None):
            return self.client.predict_batch(self.model_id, boards, encoded=encoded)

    predictor = _QueuePredictor(client, DEFAULT_MODEL_ID)
    rng = np.random.default_rng(seed)

    simulator = BatchedGameSimulator(
        predictor=predictor,
        concurrency=concurrency,
        temperature=1.0,  # replaced per game below
        max_game_moves=max_game_moves,
        model_version=model_version,
        rng=rng,
        random_open_plies=random_open_plies,
        dirichlet_epsilon=dirichlet_epsilon,
        dirichlet_alpha=dirichlet_alpha,
    )

    pending_games: List[Dict] = []
    completed = 0
    moves = 0

    def flush() -> None:
        nonlocal pending_games
        if not pending_games:
            return
        batch = {
            "white_version": model_version,
            "black_version": model_version,
            "games": pending_games,
        }
        experience_queue.put(batch)
        pending_games = []

    while completed < games_target:
        if stop_event.is_set():
            break
        local = len(simulator.completed_games) + len(pending_games)
        # Temperature decays with the global game index.
        global_index = global_start + local
        progress = min(global_index / max(temp_decay_games, 1), 1.0)
        temp = temperature + (temp_final - temperature) * progress
        simulator.agents[0].temperature = temp
        simulator.agents[1].temperature = temp

        simulator._refill()
        new = simulator.step()
        completed += new

        for game in simulator.pop_completed():
            moves += game["move_count"]
            pending_games.append(
                {
                    "result": game["result"],
                    "moves": game["move_count"],
                    "samples": game["samples"],
                }
            )
        if len(pending_games) >= flush_every:
            flush()

    flush()  # drain whatever is left
    logger.info("Worker done: games=%d moves=%d", completed, moves)


class SelfPlayWorker(Process):
    """Process wrapper around :func:`run_selfplay_worker`."""

    def __init__(
        self,
        request_queue,
        result_queue,
        experience_queue,
        games_target: int,
        global_start: int,
        games_global: int,
        concurrency: int,
        temperature: float,
        temp_final: float,
        temp_decay_games: int,
        max_game_moves: int,
        model_version: int,
        seed: int,
        stop_event,
        flush_every: int = 5,
        worker_id: int = 0,
        random_open_plies: int = 0,
        dirichlet_epsilon: float = 0.0,
        dirichlet_alpha: float = 0.03,
    ) -> None:
        super().__init__(daemon=True)
        self._args = (
            request_queue,
            result_queue,
            experience_queue,
            games_target,
            global_start,
            games_global,
            concurrency,
            temperature,
            temp_final,
            temp_decay_games,
            max_game_moves,
            model_version,
            seed,
            stop_event,
            flush_every,
            worker_id,
            random_open_plies,
            dirichlet_epsilon,
            dirichlet_alpha,
        )

    def run(self) -> None:  # pragma: no cover - thin wrapper
        run_selfplay_worker(*self._args)