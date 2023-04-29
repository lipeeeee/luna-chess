"""
    Luna-Chess Reinforcement Learning Approach
"""

# TODO: Implement a better reward function

from __future__ import annotations
import math
import random
from typing import Optional
import chess
from chess import STARTING_FEN
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class LunaRL(nn.Module):
    """Reinforcement Learning Neural Network"""

    env: object

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> None:
        """Forward prop implementation"""

class MCTS:
    """Monte-Carlo tree search Agent"""
    
    # Root node of MTCS
    root: MCTSNode
    
    # Max search iterations
    max_iterations: int

    def __init__(self, state, max_iterations=1000) -> None:
        self.root = MCTSNode(state)
        self.max_iterations = max_iterations

    def select_action(self) -> chess.Move:
        """Runs the MCTS algorithm for `self.max_iterations` iterations 
        from the root node and returns the best action to take based on the final search tree."""

        for i in range(self.max_iterations):
            # Get best action
            node = self.root.select_action()
            state = node.state
            
            # Get state reward
            if node.is_terminal():
                reward = self.get_state_reward(state)
            else:
                possible_moves = state.generate_legal_moves()
                
                if not node.children:
                    node.expand(possible_moves)
                child_node = node.select_child()
                new_state = child_node.state
                reward = self.simulate(new_state)
                
            node.update(reward)
        best_child = max(self.root.children, key=lambda child: child.visits)

        return best_child.state.peek()

    def simulate(self, state: chess.Board):
        """Simulates a game from a specified `state` using a random policy"""
        while not state.is_game_over():
            possible_moves = state.generate_legal_moves()
            move = random.choice(possible_moves)
            state = state.push(move)
        return self.get_state_reward(state)

    def get_state_reward(self, state: chess.Board) -> int:
        """Get state reward"""
        winner = state.outcome().winner
        
        return {None: 0, chess.WHITE: 1, chess.BLACK: -1}[winner]    

class MCTSNode:
    """A single node representation in MCTS"""

    # Board state
    state: chess.Board

    # Serialized board state
    __serialized_board: np.ndarray

    # The parent node in the search tree(None if root)
    parent: MCTSNode

    # List of child nodes in the search tree
    children: list[MCTSNode]

    # Sum of rewards obtained during the simulations(From node to root parent)
    total_reward: int

    # Number of times this node has been visited during MCTS
    visits: int

    # Controls the balance between exploration and exploitation.
    # A higher value of c_param encourages more exploration.
    c_param: float

    def __init__(self, state: chess.Board, parent:MCTSNode=None) -> None:
        """Initialize Node"""
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0
        self.c_param = 1.4

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
    def ucb_score(self):
        """Upper Confidence Bound(UCB) score to balance exploration and exploitation"""
        exploitation = self.total_reward / self.visits
        exploration = math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + self.c_param * exploration

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

    @property
    def is_terminal(self) -> bool:
        """Checks if current state is terminal"""
        return self.state.is_game_over()
