"""
    Luna-Chess
"""

import os
import numpy as np
import keras as keras
import chess.pgn 
from luna_board import LunaBoard
from keras import layers
from keras.models import load_model

# CHANGE THIS IF WANTING TO TRAIN A NEW MODEL OR JUST USE ANOTHER ONE
MODEL_TO_USE = "models/main_luna.h5"

class Luna():
    """Luna_chess engine main class"""

    def __init__(self, verbose=False) -> None:
        """If on initialization there is no pre-saved model we create one and train it, to then save it"""
        self.board = LunaBoard()
        self.verbose = verbose

        # Check if model exists
        if self.model_exists():
            if verbose: print(f"[BUILDING] Loading model({MODEL_TO_USE})...")
            self.model = load_model(MODEL_TO_USE)
            return
        
        # define model
        if verbose: print(f"[BUILDING] No model found, defining model...")
        # self.define(input_shape=0)

        # Main database folder(D:\dev\datasets_databases)
        self.training_folder = os.path.join("D:\\", "dev", "datasets_databases")

        # training model
        self.train()

    def define(self, input_shape) -> None:
        """Define neural net"""
        self.optimizer = 'adam'
        self.loss = 'mean_squared_error'
        
        self.model = keras.models.Sequential([
            layers.Dense(512, activation='relu', input_shape=input_shape),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='linear')
        ])

        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def train(self, N=0) -> None:
        """Train Luna on a pgn file directory(on N games)"""
        assert not self.model_exists()
    
        return


    def old_train(self) -> None:
        """Train Luna with a pgn file directory
            1. CHECK IF A MODEL DOES NOT ALREADY EXIST
            2. INITIALIZE EMPTY DATA
            3. READ ALL COMPRESSED PGN FILES IN THE TRAINING_FOLDER
            4. CONVERT DATA TO NUMPY ARRAYS
            5. DEFINE MODEL WITH NEW INPUT_SHAPE
            6. TRAIN THE MODEL
        """        
        # 1. CHECK IF A MODEL DOES NOT ALREADY EXIST
        assert not self.model_exists()

        # 2. INITIALIZE EMPTY DATA
        X_data = []
        Y_data = []

        # 3. READ ALL PGN FILES IN THE TRAINING_FOLDER
        pgn_files = os.listdir(self.training_folder)
        if self.verbose: print(f"[TRAINING] Found files: {pgn_files}")
        for pgn_file in pgn_files:
            """For each file: read all of its games into X, Y"""
           
            with open(os.path.join(self.training_folder, pgn_file), "r") as pgn:
                if self.verbose: print(f"[TRAINING] Reading file: \"{pgn}\"")
                i = 0
                while i < 200:
                    game = chess.pgn.read_game(pgn)
                    if not game:
                        break
                    
                    if self.verbose: print(f"[TRAINING] Reading game: {game}")
                    X, Y = self.preprocess_game(game)
                    X_data.extend(X)
                    Y_data.extend(Y)

                    i += 1

        # 4. CONVERT DATA TO NUMPY ARRAYS
        X_data = np.array(X_data)
        Y_data = np.array(Y_data)

        # 5. DEFINE MODEL WITH NEW INPUT_SHAPE
        print("define:", X_data.shape)
        self.define(input_shape=(X_data.shape[1],))

        print(X_data[0])
        exit(0)

        # 6. TRAIN THE MODEL
        print(f"[TRAINING] FITTING THE MODEL...")
        self.model.fit(X_data, Y_data, batch_size=128, epochs=10, validation_split=0.2)

    def preprocess_game(self, game: chess.pgn.Game):
        """Preprocess game into a vector with:
            0. Moves??
            1. Board State(8, 8)
            2. Result
            3. Pawn structure
            4. Move count
        """
        board = game.board()
        
        # Check if valid board before preprocessing
        assert board.is_valid()

        # 1. Board state encoding 
        board_state = np.zeros(64, np.uint8)
        for i in range(64):
            pp = board.piece_at(i)
            
            if pp is None:
                continue
            board_state[i] = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6, \
                    "p": 9, "n":10, "b":11, "r":12, "q":13, "k": 14}[pp.symbol()]
        
        board_state.reshape(8, 8)

        # 2. Binary state
        state = np.zeros((5, 8, 8), np.uint8)

        # 0 - 3 columns to binary
        state[0] = (board_state>>3)&1
        state[1] = (board_state>>2)&1
        state[2] = (board_state>>1)&1
        state[3] = (board_state>>0)&1

        # 4th column is who's turn it is
        state[4] = (board.turn*1.0)

        return state

    def old_preprocess_game(self, game: chess.pgn.Game):
        """Preprocess game into a vector with:
            1. Moves
            2. Board representation(8x8)
            3. Result
            4. Pawn structure
            5. Move count
        """
        
        board = game.board()
        X_moves = []
        X_board = []
        X_result = []
        X_pawn_structure = []
        X_move_count = []
        
        # Loop through all the moves in the game and process features
        for move in game.mainline_moves():
            
            # 1. Append the move to the X_moves list
            X_moves.append(move.uci())
            
            # 2. Append the board state to the X_board list
            board_state = np.zeros((8, 8, 12))
            for i in range(8):
                for j in range(8):
                    piece = board.piece_at(chess.square(i, j))
                    if piece is not None:
                        piece_type = piece.piece_type - 1
                        piece_color = int(piece.color)
                        board_state[i][j][piece_color * 6 + piece_type] = 1
            X_board.append(board_state.flatten())
            
            # 4. Append the pawn structure to the X_pawn_structure list
            pawn_structure = np.zeros((8, 8, 2))
            for i in range(8):
                for j in range(8):
                    piece = board.piece_at(chess.square(i, j))
                    if piece is not None and piece.piece_type == chess.PAWN:
                        piece_color = int(piece.color)
                        pawn_structure[i][j][piece_color] = 1
            X_pawn_structure.append(pawn_structure.flatten())
            
            # 5. Append the move count to the X_move_count list
            X_move_count.append([board.fullmove_number / 100.0])
            
            # Make the move on the board
            board.push(move)
        
        # 3. Append the result to the X_result list
        result = game.headers['Result']
        if result == '1-0':
            X_result = np.array([[1, 0, 0]] * len(X_moves))
        elif result == '0-1':
            X_result = np.array([[0, 1, 0]] * len(X_moves))
        else:
            X_result = np.array([[0, 0, 1]] * len(X_moves))

        # Combine all the features into a single array
        X = np.concatenate((np.array(X_moves).reshape(-1, 1), np.array(X_board), np.array(X_result), np.array(X_pawn_structure), np.array(X_move_count)), axis=1)
        Y = np.zeros((len(X_moves), 1))
        
        return X, Y

    def save(self) -> None:
        """Save trained model"""
        ...

    def model_exists(self) -> bool:
        """Checks if there is a pre-saved model"""
        return os.path.exists(MODEL_TO_USE)