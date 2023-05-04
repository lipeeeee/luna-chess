"""
    Debug/playground script
"""

import sys
import logging
import coloredlogs
from luna.coach import Coach
from luna.game import ChessGame as Game
from luna.NNet import Luna_Network as nn
from luna.utils import *
from luna.game.arena import Arena
from luna.mcts import MCTS
from luna.game.player import HumanChessPlayer
import numpy as np
import chess

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

args = dotdict({
    'numIters': 1000,
    'numEps': 100,                # (100)Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 10,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 100,         # Number of games moves for MCTS to simulate.
    'arenaCompare': 20,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'checkpoint': './temp/',
    'load_model': False,
    'load_examples': False,
    'load_folder_file': ('./pretrained_models/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'dir_noise': True,
    'dir_alpha': 1.4,
    'save_anyway': False        # Always save model, shouldnt be used
})

def main() -> int:
    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    log.info('Loading checkpoint "%s/"...', args.load_folder_file)
    nnet.load_checkpoint(args.checkpoint, args.load_folder_file[1])

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    log.info("Loading 'trainExamples' from file...")
    c.loadTrainExamples()

    nmcts = MCTS(g, nnet, args)
    nmcts2 = MCTS(g, nnet, args)

    def _print(x):
        print(x)

    arena = Arena(lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                  lambda x: np.argmax(nmcts2.getActionProb(x, temp=0)), g, display=_print)

    arena.playGame(verbose=True)

    return 0

if __name__ == "__main__":
    sys.exit(main())
