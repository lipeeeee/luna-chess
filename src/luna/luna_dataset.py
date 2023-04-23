"""
    Luna-Chess Pytorch dataset builder
"""

import os
import chess.pgn
import numpy as np
from torch.utils.data import Dataset
from .luna_state import LunaState
from .luna_constants import LUNA_MAIN_FOLDER, LUNA_DATA_FOLDER, LUNA_DATASET_FOLDER, LUNA_DATASET_PREFIX
from .luna_utils import *

class LunaDataset(Dataset):
    """Dataset builder for Luna"""
    
    def __init__(self, num_samples: int, verbose=False) -> None:
        self.num_samples = num_samples
        self.verbose = verbose
        
        # Main dataset folder
        self.dataset_folder = os.path.join(LUNA_MAIN_FOLDER, LUNA_DATASET_FOLDER)
        # specific dataset path given num_samples
        self.dataset_full_path = os.path.join(self.dataset_folder, LUNA_DATASET_PREFIX + str(self.num_samples) + ".npz")
        
        # if there is a dataset we load it, if not we generate one and save it
        if self.dataset_exists():
            if verbose: print(f"[DATASET] Dataset found at: {self.dataset_full_path}, loading...")
            self.load()
        else:
            if verbose: print(f"[DATASET] NO DATABASE, GENERATING AT: {self.dataset_full_path}")
            self.X, self.Y = self.generate_stockfish_dataset()
            
            if self.verbose: print(f"[DATASET] Saving dataset at: {self.dataset_full_path}...")
            self.save()

            if self.verbose: print(f"[DATASET] Loading dataset at: {self.dataset_full_path}...")
            self.load()

    def __len__(self) -> int:
        """Pytorch Dataset.len override"""
        return self.X.shape[0]

    def __getitem__(self, index) -> tuple:
        """Pytorch Dataset.getitem override"""
        return (self.X[index], self.Y[index])

    def load(self) -> None:
        """Load dataset from disk"""
        assert self.dataset_exists()

        dat = np.load(self.dataset_full_path)
        self.X = dat['arr_0']
        self.Y = dat['arr_1']

    def save(self) -> None:
        """Save dataset to disk"""
        np.savez(self.dataset_full_path, self.X, self.Y)

    # 5M Samples took: 2h10m 
    # 178_571 Samples is 1GiB
    def generate_stockfish_dataset(self, stockfish_depth=0):
        """Generate dataset with:
            X: Board serialization
            Y: Stockfish eval(depth 0 default)"""
        
        X, Y = [], []
        num_games = 0

        # Initialize stockfish .exe process
        init_stockfish()

        # Loop through every pgn file in main data folder until num_samples is met
        data_folder = os.path.join(LUNA_MAIN_FOLDER, LUNA_DATA_FOLDER)
        for fn in os.listdir(data_folder):
            pgn = open(os.path.join(data_folder, fn))
            
            # On a pgn file, read all games until none found
            while True:
                # Read game
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                board = game.board()
 
                # Read moves
                for i, move in enumerate(game.mainline_moves()):
                    # X: Serialize board
                    board.push(move)
                    ser = LunaState.serialize_board(board)
                    
                    # Y: Stockfish eval
                    sf_value = stockfish(board, stockfish_depth)
                    if not isinstance(sf_value, int):
                       continue

                    X.append(ser)
                    Y.append(sf_value)
                
                # Verbose
                if self.verbose: 
                    print(f"[DATASET] Parsing game {num_games}, got {len(X)} examples")
                    num_games += 1

                # Stop condition 
                if self.num_samples is not None and len(X) > self.num_samples:
                    X = np.array(X)
                    Y = np.array(Y)
                    return X, Y

        # Close stockfish .exe process
        close_stockfish()

        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def old_generate_dataset(self):
        """Generate dataset of N(num_samples)
            The dataset includes the board serialization(X)
            and the result of the match(Y)
        """
        X, Y = [], []
        num_games = 0
        result_dict = {"1/2-1/2": 0, "0-1": -1, "1-0": 1}

        # read pgn files in the data folder
        data_folder = os.path.join(LUNA_MAIN_FOLDER, LUNA_DATA_FOLDER)
        for fn in os.listdir(data_folder):
            pgn = open(os.path.join(data_folder, fn))
            
            # On a pgn file, read all games until none found
            while True:
                # Read Game
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                
                # Get result value(-1, 0 or 1)
                res = game.headers['Result']
                if res not in result_dict:
                    continue
                res_value = result_dict[res]

                # Append moves and result to vectors
                board = game.board()
                for i, move in enumerate(game.mainline_moves()):
                    board.push(move)
                    ser = LunaState.serialize_board(board)
                         
                    X.append(ser)
                    Y.append(res_value)
                
                # Verbose
                if self.verbose: 
                    print(f"[DATASET] Parsing game {num_games}, got {len(X)} examples")
                    num_games += 1

                # Check if we are over than the requested number of dataset samples
                if self.num_samples is not None and len(X) > self.num_samples:
                    X = np.array(X)
                    Y = np.array(Y)
                    return X, Y

        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def dataset_exists(self) -> bool:
        """Checks if a dataset with the given number of samples has been found"""
        return os.path.exists(os.path.join(LUNA_MAIN_FOLDER, LUNA_DATASET_FOLDER, self.dataset_full_path))