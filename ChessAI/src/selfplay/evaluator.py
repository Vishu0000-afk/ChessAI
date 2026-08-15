"""Model evaluation: candidate vs previous model.

Plays a configurable number of games between two models through the shared
inference server (one GPU model per *model_id*, never per worker) and reports
win/draw/loss rates plus a crude ELO estimate. The coordinator uses this to
decide whether to promote a freshly trained candidate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import chess
import numpy as np

from src.learning.encoding import move_to_index
from src.selfplay.inference import InferenceClient

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Outcome of an evaluation run."""

    games: int
    a_wins: int
    b_wins: int
    draws: int
    score: float  # a's score (wins + 0.5 * draws) / games
    elo_estimate: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "games": self.games,
            "a_wins": self.a_wins,
            "b_wins": self.b_wins,
            "draws": self.draws,
            "score": self.score,
            "elo_estimate": self.elo_estimate,
        }


class _IdPredictor:
    def __init__(self, client: InferenceClient, model_id: str) -> None:
        self.client = client
        self.model_id = model_id

    def predict_batch(self, boards):
        return self.client.predict_batch(self.model_id, boards)


def evaluate_models(
    request_queue,
    result_queue,
    model_a: str,
    model_b: str,
    num_games: int,
    concurrency: int = 8,
    max_game_moves: int = 400,
) -> EvalResult:
    """Play model_a vs model_b (colors alternate per game) and score results.

    Moves are deterministic (argmax over the masked policy). Returns the
    candidate ``a``'s win/draw/loss counts and an ELO estimate.
    """
    client = InferenceClient(request_queue, result_queue)
    predictor_a = _IdPredictor(client, model_a)
    predictor_b = _IdPredictor(client, model_b)

    games: List[Dict] = []
    completed = 0
    a_wins = 0
    b_wins = 0
    draws = 0

    def new_game() -> Dict:
        is_a_white = len(games) % 2 == 0
        return {
            "board": chess.Board(),
            "a_white": is_a_white,
            "moves": 0,
        }

    while completed < num_games:
        while len(games) < concurrency:
            games.append(new_game())

        # Group the current positions by the model of the side to move.
        for a_group in (True, False):
            subset = [g for g in games if (g["board"].turn == chess.WHITE) == (g["a_white"] == a_group)]
            if not subset:
                continue
            boards = [g["board"] for g in subset]
            model_id = model_a if a_group else model_b
            logits, _ = predictor_a.predict_batch(boards) if model_id == model_a else predictor_b.predict_batch(boards)

            for i, g in enumerate(subset):
                move = _argmax_move(g["board"], logits[i])
                if move is None:
                    continue
                g["board"].push(move)
                g["moves"] += 1

        # Score and retire finished games.
        still_active = []
        for g in games:
            if g["board"].is_game_over() or g["moves"] >= max_game_moves:
                result = g["board"].result(claim_draw=True) if g["board"].is_game_over() else "1/2-1/2"
                a_side = g["a_white"]
                if result == "1/2-1/2":
                    draws += 1
                elif (result == "1-0") == a_side:
                    a_wins += 1
                else:
                    b_wins += 1
                completed += 1
            else:
                still_active.append(g)
        games = still_active

    total = max(completed, 1)
    score = (a_wins + 0.5 * draws) / total
    # ELO from expected score: 1/(1+10^(-elo/400)) = score.
    score_c = min(max(score, 0.001), 0.999)
    elo = -400.0 * np.log10((1.0 - score_c) / score_c)
    return EvalResult(
        games=completed,
        a_wins=a_wins,
        b_wins=b_wins,
        draws=draws,
        score=float(score),
        elo_estimate=float(elo),
    )


def _argmax_move(board: chess.Board, logits: np.ndarray) -> Optional[chess.Move]:
    moves = list(board.legal_moves)
    if not moves:
        return None
    best = None
    best_val = -1e18
    for m in moves:
        v = float(logits[move_to_index(m, board.turn)])
        if v > best_val:
            best_val = v
            best = m
    return best