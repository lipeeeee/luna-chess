"""
    Luna-Chess, a chess engine rated around X

    Project Architecture    
    Wrapper(either html or anything else) ->
        Luna ->
            Luna_Utils ->
            Luna_State ->
            Luna_Eval ->
                Luna_NN ->
                Luna_dataset ->

    by lipeeeee
"""
import sys
import logging
import coloredlogs
from luna.coach import Coach
from luna.game import ChessGame as Game
from luna.NNet import Luna_Network as nn
from luna.luna_utils import *

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 1,                # (100)Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 10,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 100,         # Number of games moves for MCTS to simulate.
    'arenaCompare': 20,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./pretrained_models/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'dir_noise': True,
    'dir_alpha': 1.4,
})

def main():
    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process')
    c.learn()

if __name__ == "__main__":
    sys.exit(main())