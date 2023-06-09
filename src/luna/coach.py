"""
    Coach for the self-play learning
"""

import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import numpy as np
from tqdm import tqdm

# Luna
from .game.arena import Arena
from .mcts import MCTS
from .game.luna_game import ChessGame
from .NNet import Luna_Network 
from .utils import dotdict

log = logging.getLogger(__name__)

class Coach():
    """
        This class executes the self-play + learning. It uses the functions defined
        in Game and NeuralNet. args are specified in main.py.
    """

    # Game Environment
    game: ChessGame

    # Neural net Wrapper
    nnet: Luna_Network
    pnet: Luna_Network # Competitor Network

    # dotdict arguments
    args: dotdict

    # MCTS algorithm
    mcts: MCTS

    # history of examples from args.numItersForTrainExamplesHistory latest iterations
    trainExamplesHistory: list

    # can be overriden in loadTrainExamples()
    skipFirstSelfPlay: bool

    def __init__(self, game, nnet, args) -> None:
        super(Coach, self).__init__() 

        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game) 
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False

    def executeEpisode(self) -> list[tuple]:
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi, v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        # Play game until is over
        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            bs, ps = zip(*self.game.getSymmetries(canonicalBoard, pi))
            _, valids_sym = zip(*self.game.getSymmetries(canonicalBoard, valids))
            sym = zip(bs,ps,valids_sym)

            for b, p, valid in sym:
                trainExamples.append([self.game.toArray(b), self.curPlayer, p, valid])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)
            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer)), x[3]) for x in trainExamples]

    def learn(self) -> None:
        """
        Performs `numIters` iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            log.info(f'Starting Iter #{i} ...')

            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                # Perform MCTS to get self-play data
                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            pmcts = MCTS(self.game, self.pnet, self.args)
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)

            # Each network will play with white and black
            num_arena_pits = int(self.args.arenaCompare / 2)
            oneWon = 0
            twoWon = 0
            draws = 0

            # First Pit
            for _ in tqdm(range(num_arena_pits), desc="Arena.playGames (1)"):
                gameResult = arena.playGame(verbose=False)
                if gameResult == 1:
                    oneWon += 1
                elif gameResult == -1:
                    twoWon += 1
                else:
                    draws += 1
                pmcts = MCTS(self.game, self.pnet, self.args)
                nmcts = MCTS(self.game, self.nnet, self.args)

            arena = Arena(lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(pmcts.getActionProb(x, temp=0)), self.game)

            # Second Pit, changed colors
            for _ in tqdm(range(num_arena_pits), desc="Arena.playGames (2)"):
                gameResult = arena.playGame(verbose=False)
                if gameResult == -1:
                    oneWon += 1
                elif gameResult == 1:
                    twoWon += 1
                else:
                    draws += 1
                pmcts = MCTS(self.game, self.pnet, self.args)
                nmcts = MCTS(self.game, self.nnet, self.args)

            pwins = oneWon
            nwins = twoWon

            # Compare models
            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if self.args.save_anyway:
                log.warning("NOT CHECKING MODEL'S PERFORMANCE(args.save_anyway=True)")
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            else:
                if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                    log.info('REJECTING NEW MODEL')
                    self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                else:
                    log.info('ACCEPTING NEW MODEL')
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration) -> str:
        """We save checkpoints based on an iteration"""
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration) -> None:
        """Save training examples from MCTS search"""
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self) -> None:
        """Load training examples of MCTS search from disk"""
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
