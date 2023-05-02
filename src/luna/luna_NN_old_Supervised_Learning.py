"""
    - DEPRECATED -
    
    Luna-Chess artificial neural network
    Supervised Learning
"""

import os
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .luna_constants import LUNA_MAIN_FOLDER, CURRENT_MODEL, NUM_SAMPLES, LUNA_MODEL_FOLDER, CUDA
from .luna_dataset import LunaDataset

# Note: Training only has a CUDA implementation
class LunaNN_old(nn.Module):
    """Pytorch Neural Network"""

    def __init__(self, model_file=CURRENT_MODEL, verbose=False, epochs=100, save_after_each_epoch=False) -> None:
        super(LunaNN_old, self).__init__()

        # Neural Net definition
        if verbose: print(f"[NEURAL NET] Initializing neural network...")
        self.define()
        self.lr = 1e-3
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.batch_size = 256
        self.epochs = epochs
        self.save_after_each_epoch = save_after_each_epoch
        
        # Cuda handling, only CUDA training for now
        if CUDA:
            self.cuda()
        else:
            raise Exception("Non-Cuda implementation still TODO")
        
        self.model_file = model_file # .pt file(ex: main_luna.pt)
        self.model_path = os.path.join(LUNA_MAIN_FOLDER, LUNA_MODEL_FOLDER, model_file)
 
        # Check if existing model
        if self.model_exists():
            if verbose: print(f"[NEURAL NET] Found existing model at: {self.model_path}, loading...")
            self.load()
        else:            
            if verbose: print(f"[NEURAL NET] NO EXISTING NEURAL NET AT: {self.model_path}, Training new neural network...")

            # Dataset Initialazation
            if verbose: print(f"[DATASET] Initializing dataset...")
            self.dataset = LunaDataset(num_samples=NUM_SAMPLES, verbose=verbose)    
            self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)        

            if verbose: print(f"[NEURAL NET] Training model...")
            self._train(epochs=self.epochs, save_after_each_epoch=self.save_after_each_epoch)

            if verbose: print(f"[NEURAL NET] FINISHED TRAINING, SAVING...")
            self.save()

    def define(self) -> None:
        """Define Net
            Net results after 39EPOCHS(16h) training, 24m each:
                -> last loss: EPOCH[39]: 4161.70907 in 24min
                -> avg stockfish diff: 7.59
                -> rating: X
                -> size: 22MB
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
        return self.last(x)

    """luna-first
    def define(self) -> None:
        Define Net
            Net results after 35EPOCHS(14h) training:
                -> last loss: EPOCH [ 35]: 8512.545898
                -> avg stockfish diff: 41.17
                -> rating: X
                -> size: 140MB        
        # input
        self.conv1 = nn.Conv2d(24, 64, kernel_size=3, padding=1)
        
        # ConvNets
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

        # Dense
        self.fc1 = nn.Linear(2048, 1024)
        self.droupout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(1024, 512) 
        self.droupout2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(512, 256)

        # output
        self.last = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        Forward Prop
        #input
        x = self.conv1(x)
        
        # conv2
        x = F.relu(self.bn2(self.conv2(x)))
        
        # conv3
        x = F.relu(self.pool(self.bn3(self.conv3(x))))

        # conv4
        x = F.relu(self.pool(self.bn4(self.conv4(x))))

        # reshape to fc    
        x = x.view(x.size(0), -1)
        
        # Dense layers
        # fc1
        x = F.relu(self.fc1(x))
        x = self.droupout1(x)
        
        # fc2
        x = F.relu(self.fc2(x))
        x = self.droupout2(x)

        # fc3
        x = F.relu(self.fc3(x))

        return self.last(x)
    """    


    def lite_define(self) -> None:
        """Define neural net"""
        
        # input
        self.conv1 = nn.Conv2d(15, 32, kernel_size=3, padding=1)

        # hidden
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(1024 * 2 * 2, 2048)
        #self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(2048, 1024)
        #self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(1024, 512)
        #self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(512, 1)

    def lite_forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))

        x = x.view(-1, 1024 * 2 * 2)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    def load(self) -> None:
        """Load luna from a .pth file"""
        self.load_state_dict(torch.load(self.model_path))
        self.eval()

    def save(self) -> None:
        """Save luna weights and biases and everything else into a .pt file"""
        torch.save(self.state_dict(), self.model_path)
    
    def _train(self, epochs, save_after_each_epoch=True) -> None:
        """Train LunaNN()(only on cuda)"""
        assert not self.model_exists()
        assert CUDA
        
        self.train()

        for epoch in range(epochs):
            epoch_clock = time.time()
            all_loss = 0
            num_loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                target = target.unsqueeze(-1)
                data, target = data.to("cuda"), target.to("cuda")
                data = data.float()
                target = target.float()

                self.optimizer.zero_grad()
                output = self(data)

                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()
                
                all_loss += loss.item()
                num_loss += 1

            print(f"EPOCH[{epoch}]: {all_loss/num_loss} in {(time.time() - epoch_clock)/60}min")

            if save_after_each_epoch: self.save()

    def model_exists(self) -> bool:
        """Checks if there is a pre-saved model"""
        return os.path.exists(self.model_path)