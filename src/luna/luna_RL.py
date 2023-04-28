"""
    Luna-Chess Reinforcement Learning Approach
"""

from __future__ import annotations
import math
import chess
import numpy as np
from torch import nn
from torch.nn import functional as F

class LunaRL(nn.Module):
    """Reinforcement Learning Neural Network"""

    env: object

    def __init__(self) -> None:
        super()

    def forward() -> None:
        """Forward prop implementation"""

class MCTS:
    """Monte-Carlo tree search algorithm"""

    def __init__(self) -> None:
        ...

class MCTSNode:
    """A single node representation in MCTS"""

    # Board state
    state: chess.Board

    # Serialized board state
    __serialized_board: np.ndarray

    # The parent node in the search tree(None if root)
    parent: MCTSNode

    # List of child nodes in the search tree
    children: list

    # Sum of rewards obtained during the simulations
    total_reward: int

    # Number of times this node has been visited during MCTS
    visits: int

    def __init__(self, state: chess.Board, parent:MCTSNode=None) -> None:
        """Initialize Node"""
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0

    # TODO: double_check if python's variable referencing logic doenst
    # mess this up
    def expand(self, possible_moves: list[chess.Move]) -> None:
        """Expand Node tree"""
        for move in possible_moves:
            # Apply a move
            new_state = self.state.copy()
            new_state.push(move)

            # Create Node
            new_node = MCTSNode(new_state, parent=self)

            # Children
            self.children.append(new_node)

    @property
    def ucb_score(self, c_param=1.4):
        """Upper Confidence Bound(UCB) score to balance exploration and exploitation"""
        exploitation = self.total_reward / self.visits
        exploration = math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + c_param * exploration

    def select_child(self) -> MCTSNode:
        """Returns highest UCB score child"""
        return max(self.children, key=lambda child: child.ucb_score())

    def update(self, reward:int) -> None:
        """Updates visits and total reward count of this node and its parents"""
        self.visits += 1
        self.total_reward += reward
        
        if self.parent:
            self.parent.update(reward) 

    def select_action(self):
        """Best action/node to follow given current search tree"""
        simulation_node = self
        
        while simulation_node.children:
            simulation_node = simulation_node.select_child()
        
        return simulation_node

    @property
    def serialized_board(self) -> np.ndarray:
        """Calculate serialized board"""
        return self.__serialized_board
