"""Self-play package: parallel AI-vs-AI games with continuous learning."""

from src.selfplay.benchmark import run_benchmark
from src.selfplay.coordinator import SelfPlayCoordinator
from src.selfplay.evaluator import EvalResult, evaluate_models
from src.selfplay.inference import InferenceClient, InferenceServer
from src.selfplay.statistics import Dashboard, SelfPlayStats
from src.selfplay.worker import BatchedGameSimulator, SelfPlayWorker

__all__ = [
    "run_benchmark",
    "SelfPlayCoordinator",
    "EvalResult",
    "evaluate_models",
    "InferenceClient",
    "InferenceServer",
    "Dashboard",
    "SelfPlayStats",
    "BatchedGameSimulator",
    "SelfPlayWorker",
]