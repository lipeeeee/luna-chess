"""
    Neural Network Wrapper
"""

import os
import time
import numpy as np
import logging
import torch
import torch.optim as optim
from .luna_NN import LunaNN as net
from .utils import dotdict, AverageMeter
from .game.luna_game import ChessGame

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
log = logging.getLogger(__name__)

# Hyper Params
args = dotdict({
    'lr': 0.02,
    'dropout': 0.3,
    'epochs': 20,
    'batch_size': 64,
    'cuda': True,
    'num_channels': 128,
})

class Luna_Network(object):
    """Main Neural Network Class"""

    # Neural Net Architecture
    nnet: net

    # Board Dimensions
    board_x: int
    board_y: int
    board_z: int

    # Action Size
    action_size: int

    def __init__(self, game: ChessGame) -> None:
        super(Luna_Network, self).__init__()

        self.nnet = net(game, args)
        self.board_x, self.board_y, self.board_z = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.nnet.cuda()

    def train(self, examples) -> None:
        """
            Train on `examples`
            
            Args:
                examples: list of examples, each example is of form 
                (board, pi, v, valids)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch+1))
            self.nnet.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            batch_idx = 0

            while batch_idx < int(len(examples) / args.batch_size):
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs, valids = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
                target_valids = torch.FloatTensor(np.array(valids))

                # Cuda performance improvement
                boards, target_pis, target_vs, target_valids = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda(), target_valids.contiguous().cuda()

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                out_pi, out_v = self.nnet((boards, target_valids))
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                writer.add_scalar("Loss/train", l_pi.item(), batch_idx)
                writer.add_scalar("Loss/train", l_v.item(), batch_idx)
                writer.flush()
                
                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                log.info('({epoch}: {batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f} | Total Loss: {tl:.4f}'.format(
                            batch=batch_idx,
                            size=int(len(examples)/args.batch_size),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            lpi=pi_losses.avg,
                            lv=v_losses.avg,
                            tl=total_loss,
                            epoch=epoch+1
                            ))

    def predict(self, boardAndValid) -> tuple:
        """
            Given a board, predicts probabilty distribuition and scalar board value

            Args:
                boardAndValid: board
        """
        # timing
        board, valid = boardAndValid

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        valid = torch.FloatTensor(valid.astype(np.float64))
        if args.cuda:
            board = board.contiguous().cuda()
            valid = valid.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y, self.board_z)
        
        # predict without changing weigths
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet((board, valid))

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        """Custom loss function for probabilty distribuition"""
        return -torch.sum(targets*outputs)/targets.size()[0]

    def loss_v(self, targets, outputs):
        """Custom loss function for scalar value"""
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]

    def loss(self, l_pi: torch.Tensor, l_v: torch.Tensor) -> torch.Tensor: 
        """Loss function
            l=∑t(vθ(st)-zt)2-→πt⋅log(→pθ(st))
        """

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar') -> None:
        """Save weights checkpoint"""
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")

        torch.save({
            'state_dict' : self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar') -> None:
        """Load Weights"""
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))

        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

    def print(self, game) -> None:
        """Print current self object state"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(net(game, args).to(device))
