"""
    Luna-Chess Reinforcement Neural Network Architecture
    
    NOTE
    Neural Network
    - Inputs:
        . b - serialized board
    - Outputs:
        . vθ(s) - a scalar value of the board state ∈ [-1,1] from the perspective of the current player
        . →pθ(s) - a policy that is a probability vector over all possible actions.

    Training
    (st,→πt,zt), where:
        - st is the state
        - →πt is an estimate of the probability from state st
        - zt final game outcome ∈ [-1,1]

    Loss function:
    l=∑t(vθ(st)-zt)2-→πt⋅log(→pθ(st))
"""

from __future__ import annotations
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from .game.luna_game import ChessGame
from .luna_utils import dotdict

class LunaNN(nn.Module):
    """Reinforcement Learning Neural Network"""

    # Optimizer
    optimizer: optim.Optimizer

    # Learning Rate
    learning_rate: float

    # Action size
    action_size: int

    # Game instance
    game: ChessGame

    # HyperParameter args
    args: dotdict

    def __init__(self, game: ChessGame, args: dotdict) -> None:
        super(LunaNN, self).__init__()

        self.board_x, self.board_y, self.board_z = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.game = game
        self.args = args

        # Define neural net
        self.define_architecture()
        self.learning_rate = 1e-3
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def define_architecture(self) -> None:
        """Define Net
            - Input: serialized chess.Board
            - Output:
                - predicted board value (tanh)
                - policy distribution over possible moves (softmax)
        """
        # Args shortcut
        args = self.args

        # Input
        self.conv1 = nn.Conv3d(1, args.num_channels, 3, stride=1, padding=1)
        
        ## Hidden
        self.conv2 = nn.Conv3d(args.num_channels, args.num_channels * 2, 3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(args.num_channels * 2, args.num_channels * 2, 3, stride=1)
        self.conv4 = nn.Conv3d(args.num_channels * 2, args.num_channels * 2, 3, stride=1)
        self.conv5 = nn.Conv3d(args.num_channels * 2, args.num_channels * 2, 1, stride=1)

        self.bn1 = nn.BatchNorm3d(args.num_channels)
        self.bn2 = nn.BatchNorm3d(args.num_channels * 2)
        self.bn3 = nn.BatchNorm3d(args.num_channels * 2)
        self.bn4 = nn.BatchNorm3d(args.num_channels * 2)
        self.bn5 = nn.BatchNorm3d(args.num_channels * 2)

        self.fc1 = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4)*(self.board_z-4), 1024) #4096 -> 1024
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, 512)
        self.fc_bn3 = nn.BatchNorm1d(512)

        # output p dist        
        self.fc4 = nn.Linear(512, self.action_size)

        # output scalar
        self.fc5 = nn.Linear(512, 1)

    def forward(self, boardsAndValids):
        """Forward prop"""
        x, valids = boardsAndValids

        x = x.view(-1, 1, self.board_x, self.board_y, self.board_z)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4)*(self.board_z-4))

        x = F.dropout(F.relu(self.fc_bn1(self.fc1(x))), p=self.args.dropout, training=self.training)
        x = F.dropout(F.relu(self.fc_bn2(self.fc2(x))), p=self.args.dropout, training=self.training)
        x = F.dropout(F.relu(self.fc_bn3(self.fc3(x))), p=self.args.dropout, training=self.training)

        pi = self.fc4(x)
        v = self.fc5(x)

        pi -= (1 - valids) * 1000
        return F.log_softmax(pi, dim=1), torch.tanh(v)
