"""
    Luna-Chess Reinforcement Learning Approach

    NOTE: 
        -> The true_values variable in the AlphaZero implementation represents the actual value of
        a state as determined by the final outcome of a game played from that state(USING RANDOM SIMULATION)
        -> 
    
    TODO:
    - Create a neural network class in PyTorch that takes in the game state as input and outputs a 
    predicted value of the state.

    - Initialize a neural network instance as a value network.

    During the MCTS algorithm, for each node in the tree, 
    evaluate the state with the value network to obtain a predicted value. 
    Use this predicted value to backpropagate the result of the simulation. 
    This will allow the network to learn to predict the expected outcome of each state.

    Train the value network using the Monte Carlo estimates of the true values that were obtained 
    during the backpropagation step. The goal is to minimize the mean squared error between the predicted 
    value and the Monte Carlo estimate of the true value.
"""

from __future__ import annotations
import math
import random
import chess
import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch import optim
from torch.nn import functional as F
from luna_state import LunaState

class LunaRL(nn.Module):
    """Reinforcement Learning Value Neural Network"""

    # Optimizer
    optimizer: optim.Optimizer

    # Learning Rate
    learning_rate: float

    # Loss fn
    loss: _Loss

    def __init__(self) -> None:
        super().__init__()

        # Define neural net
        self.define_architecture()
        self.learning_rate = 1e-3
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()

    def define_architecture(self) -> None:
        """Define Net
            - Input: 24x8x8 (serialized chess.Board)
            - Output: predicted board value(tanh)
        """        
        ### Input 24, 8, 8
        self.conv1 = nn.Conv2d(24, 64, kernel_size=3, padding=1)
        
        ### Hidden
        # conv2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # conv3
        self.conv3 = nn.Conv2d(128, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)

        # conv4
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # fc1
        self.fc1 = nn.Linear(2048, 1024)
        self.droupout1 = nn.Dropout(p=0.5)

        # fc2
        self.fc2 = nn.Linear(1024, 512) 
        self.droupout2 = nn.Dropout(p=0.5)

        # fc3
        self.fc3 = nn.Linear(512, 256)

        ### Output
        self.last = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        """Forward Prop"""
        ### Input 24, 8, 8
        x = self.conv1(x)

        ### Hidden        
        # conv2
        x = F.relu(self.bn2(self.conv2(x)))
        
        # conv3
        x = F.relu(self.pool(self.bn3(self.conv3(x))))

        # conv4
        x = F.relu(self.pool(self.bn4(self.conv4(x))))

        # reshape to fc1 (2048)    
        x = x.view(x.size(0), -1)
        
        # fc1
        x = F.relu(self.fc1(x))
        x = self.droupout1(x)
        
        # fc2
        x = F.relu(self.fc2(x))
        x = self.droupout2(x)

        # fc3
        x = F.relu(self.fc3(x))

        ### Output
        return F.tanh(self.last(x)) # [-1 to 1]

    def _train() -> None:
        """Train value-network"""

class MCTS:
    """Monte-Carlo tree search Agent"""

    # Root node of MTCS
    root: MCTSNode

    # Value Network
    value_network: LunaRL

    # Max search iterations
    max_iterations: int

    def __init__(self, state: chess.Board, value_network: LunaRL, max_iterations=1000) -> None:
        self.root = MCTSNode(state)
        self.value_network = value_network
        self.max_iterations = max_iterations

    def select_action(self) -> chess.Move:
        """Runs the MCTS algorithm for `self.max_iterations` iterations 
        from the root node and returns the best action to take based on the final search tree."""

        for _ in range(self.max_iterations):
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

    def get_state_reward(self, state: chess.Board) -> float:
        """Get state reward, predict if board is not over"""
        value: float

        # If game is not over predict state reward
        if not state.is_game_over():
            with torch.no_grad():
                # Get board ready for input
                serialized_board = LunaState.serialize_board(state)
                state_tensor = torch.Tensor(serialized_board)

                # Input to neural net
                value = self.value_network(state_tensor)

                # Result
                value = value.item()            
        else:
            winner = state.outcome().winner
            value = {None: 0.0, chess.WHITE: 1.0, chess.BLACK: -1.0}[winner]
 
        return value
            
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
        self.c_param = 1 # alphazero value= 1

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
        """Highes UCB score child to follow given current search tree"""
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
