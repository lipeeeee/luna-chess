"""
    Luna-Chess neural network
"""

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .luna_constants import LUNA_MAIN_FOLDER, CURRENT_MODEL, NUM_SAMPLES, LUNA_MODEL_FOLDER, CUDA
from .luna_dataset import LunaDataset

# Note: Training only has a CUDA implementation
class LunaNN(nn.Module):
    """Pytorch Neural Network"""

    def __init__(self, model_file=CURRENT_MODEL, verbose=False, epochs=100, save_after_each_epoch=False) -> None:
        super(LunaNN, self).__init__()

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

        # if there isnt a model generate dataset
        if not self.model_exists():
            # Dataset Initialazation
            if verbose: print(f"[DATASET] Initializing dataset...")
            self.dataset = LunaDataset(num_samples=NUM_SAMPLES, verbose=verbose)    
            self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)        
        
        # Check if existing model
        if self.model_exists():
            if verbose: print(f"[NEURAL NET] Found existing model at: {self.model_path}, loading...")
            self.load()
        else:
            if verbose: print(f"[NEURAL NET] NO EXISTING NEURAL NET AT: {self.model_path}, Training new neural network...")
            self._train(epochs=self.epochs, save_after_each_epoch=self.save_after_each_epoch)

            if verbose: print(f"[NEURAL NET] FINISHED TRAINING, SAVING...")
            self.save()

    def define(self) -> None:
        """Define Net"""

        # input
        self.conv1 = nn.Conv2d(24, 32, kernel_size=3, padding=1)

        # hidden
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=5, padding=2)
        self.bn6 = nn.BatchNorm2d(1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(1024 * 2 * 2, 2048)
        self.bn7 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn8 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(1024, 512)
        self.bn9 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(512, 1)

        self.shortcut2 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.shortcut3 = nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.shortcut4 = nn.Conv2d(128, 256, kernel_size=1, stride=1)
        self.shortcut5 = nn.Conv2d(256, 512, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        shortcut2 = self.shortcut2(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x) + shortcut2))
        
        shortcut3 = self.shortcut3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x) + shortcut3))
        
        shortcut4 = self.shortcut4(x)
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.pool(self.bn5(self.conv5(x) + shortcut4)))
        
        x = x.view(-1, 1024 * 2 * 2)
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn8(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn9(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)

        return x

    def decent_define(self) -> None:
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

    def decent_forward(self, x: torch.Tensor):
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

    def old_define(self) -> None:
        """Define neural net"""
        # input
        self.conv1 = nn.Conv2d(15, 32, kernel_size=3, padding=1)

        # hidden
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # output
        self.fc4 = nn.Linear(128, 1)

    def old_forward(self, x:torch.Tensor):
        """Forward prop"""
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = x.view(-1, 256 * 2 * 2)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        x = self.fc4(x)
        return x

    def old_define(self) -> None:
        """Define neural net"""
        self.a1 = nn.Conv2d(15, 16, kernel_size=3, padding=1)
        self.a2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.a3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)

        self.b1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        self.c1 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c2 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c3 = nn.Conv2d(64, 128, kernel_size=2, stride=2)

        self.d1 = nn.Conv2d(128, 128, kernel_size=1)
        self.d2 = nn.Conv2d(128, 128, kernel_size=1)
        self.d3 = nn.Conv2d(128, 128, kernel_size=1)

        self.last = nn.Linear(128, 1)

    def old_forward(self, x: torch.Tensor):
        """Forward prop implementation"""
        x = F.relu(self.a1(x))
        x = F.relu(self.a2(x))
        x = F.relu(self.a3(x))

        # 4x4
        x = F.relu(self.b1(x))
        x = F.relu(self.b2(x))
        x = F.relu(self.b3(x))

        # 2x2
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))

        # 1x128
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))

        x = x.view(-1, 128)
        x = self.last(x)

        # value output
        return torch.tanh(x)

    def secondary_define(self) -> None:
        """Secondary neural network"""
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=256 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=1)

    def secondary_forward(self, x):
        """Secondary forward prop implementation for secondary_deifne"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def load(self) -> None:
        """Load luna from a .pth file"""
        self.load_state_dict(torch.load(self.model_path))
        # self.eval()

    def save(self) -> None:
        """Save luna weights and biases and everything else into a .pt file"""
        torch.save(self.state_dict(), self.model_path)
    
    def _train(self, epochs, save_after_each_epoch=True) -> None:
        """Train LunaNN()(only on cuda)"""
        assert not self.model_exists()
        assert CUDA
        
        self.train()

        for epoch in range(epochs):
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

            print("EPOCH [%3d]: %f" % (epoch, all_loss/num_loss))
            if save_after_each_epoch: self.save()

    def model_exists(self) -> bool:
        """Checks if there is a pre-saved model"""
        return os.path.exists(self.model_path)