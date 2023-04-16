"""
    Luna-Chess
"""

import os
from luna_board import LunaBoard
import chess.pgn 
import numpy as np
import keras as keras
from keras import layers
from keras.models import load_model

# CHANGE THIS IF WANTING TO TRAIN A NEW MODEL OR JUST USE ANOTHER ONE
MODEL_TO_USE = "models/main_luna.h5"

class Luna():
    """Luna_chess engine main class"""

    def __init__(self) -> None:
        """If on initialization there is no pre-saved model we create one and train it, to then save it"""
        self.board = LunaBoard()
        
        # Check if model exists
        if self.model_exists():
            self.model = load_model(MODEL_TO_USE)
            return
        
        # define model
        self.optimizer = 'adam'
        self.loss = 'mean_squared_error'
        self.N = 864
        self.input_shape = (8, 8, 12) # optimize input shape!!!
        
        self.define()

        # Main database folder(D:\dev\datasets_databases)
        self.training_folder = os.path.join("D", "dev", "datasets_databases")

    def define(self) -> None:
        """Define neural net"""
        self.model = keras.models.Sequential([
            layers.Conv2D(64, 3, activation='relu', input_shape=self.input_shape),
            layers.Conv2D(64, 3, activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')
        ])

        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def train(self, pgn_file) -> None:
        """Train Luna with a pgn file
            TODO: in future make it so it supports multiple pgn files
        """        
        assert not self.model_exists()

        X_data = []
        Y_data = []

        # Read all games
        pgn = os.open(os.path.join(self.training_folder, pgn_file))
        while True:
            game = chess.pgn.read_game(pgn)
            if not game:
                break

            X, Y = self.preprocess_game(game)
            X_data.extend(X)
            Y_data.extend(Y)

        # Convert the data to numpy arrays
        X_data = np.array(X_data)
        Y_data = np.array(Y_data)

        print(f"Training Luna on {pgn}...")

    def preprocess_game(self, game: chess.pgn.Game):
        """So much work to be done!"""
        
        board = game.board()
        X = []
        Y = []

        for move in game.mainline_moves():
            x = np.zeros(self.input_shape)
            move_san = board.san(move)
            from_square = move.from_square
            to_square = move.to_square
            x[:,:,board.piece_type_at(from_square) - 1] = 1
            x[:,:,6 + board.piece_type_at(to_square) - 1] = 1
            X.append(x)
            Y.append(move_san)
            board.push(move)

        return X, Y

    def model_exists(self) -> bool:
        """Checks if there is a pre-saved model"""
        return os.path.exists(MODEL_TO_USE)