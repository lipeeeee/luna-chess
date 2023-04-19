"""
    Luna-Chess Pytorch dataset builder
"""

import os
import chess.pgn
import numpy as np
from torch.utils.data import Dataset

from .luna_constants import LUNA_MAIN_FOLDER, LUNA_DATA_FOLDER, LUNA_DATASET_FOLDER, LUNA_DATASET_PREFIX

class LunaDataset(Dataset):
    """Dataset builder for Luna"""

    def __init__(self, num_samples, verbose=False) -> None:
        self.num_samples = num_samples
        self.verbose = verbose

        self.dataset_folder = os.path.join(LUNA_MAIN_FOLDER, LUNA_DATASET_FOLDER)
        self.dataset_full_path = os.path.join(self.dataset_folder, LUNA_DATASET_PREFIX + str(self.num_samples) + ".npz")

        if self.dataset_exists():
            if verbose: print(f"[DATASET] Dataset found at: {self.dataset_full_path}, loading...")
            self.load()
        else:
            if verbose: print(f"[DATASET] NO DATABASE, GENERATING AT: {self.dataset_full_path}...")
            self.X, self.Y = self.generate_dataset()
            
            if self.verbose: print(f"[DATASET] Saving dataset at: {self.dataset_full_path}...")
            self.save()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index) -> tuple:
        return (self.X[index], self.Y[index])

    def load(self) -> None:
        """Load dataset from disk"""
        assert self.dataset_exists()

        dat = np.load(self.dataset_full_path)
        self.X = dat['arr_0']
        self.Y = dat['arr_1']

    def save(self) -> None:
        """Save dataset"""
        np.savez(self.dataset_full_path, self.X, self.Y)

    def generate_dataset(self):
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
                    ser = self.serialize_board(board)
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