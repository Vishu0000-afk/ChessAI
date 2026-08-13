"""Agent implementations (game-facing decision makers).

The game infrastructure only depends on the abstract ``ChessAgent``
interface, so bots can be mixed and matched freely:

    Bot A = NeuralAgent
    Bot B = NeuralAgent

or, later:

    Bot A = CurrentModel
    Bot B = PreviousModel
    Bot A = NeuralAgent
    Bot B = ClassicalEngineAgent
"""

from src.agents.base import ChessAgent
from src.agents.classical_engine import ClassicalEngineAgent
from src.agents.neural_agent import NeuralAgent
from src.agents.random_agent import RandomAgent

__all__ = [
    "ChessAgent",
    "ClassicalEngineAgent",
    "NeuralAgent",
    "RandomAgent",
]