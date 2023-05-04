<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/github/license/R3nzTheCodeGOD/R3nzSkin.svg?style=for-the-badge)

# Luna Chess
</div>
Luna-Chess is a single-thread single-GPU chess engine rated around 1850, It trains a chess engine through pure self-play without <i>any</i> human knowledge except the rules of the game.

![Sem título](https://user-images.githubusercontent.com/62669782/233196743-ed90f2c3-7e2d-4a42-a469-b344e99115a4.png)


<p>

<p>

## Deep Neural Network
The neural network **f0** is parameterised by **0** and takes input the state **s** of the board. It has two outputs: a continuous value/evaluation of the board state **vθ(s)∈[−1,1]** from the prespective of the current player, and a policy **pθ(s)** that is a probability vector over all possible actions.

When training the network, at the end of each game of self-play, the neural net is provided training examples of the form **(st, πt, zt)**. **πt** is an esitmate of the policy from state **st** and **zt∈{−1,1}** is the final outcome of the game from the perspective of the player at **st**.

The neural net is trained to minimise the following loss function:

![image](https://user-images.githubusercontent.com/62669782/236341753-92420ce8-1636-46f3-900f-0d2407d1c38e.png)


The idea is that, over time, the network will learn what states eventually lead to wins or losses.

### Architecture
```python
self.conv1 = nn.Conv3d(1, args.num_channels, 3, stride=1, padding=1)
        
## Hidden
self.conv2 = nn.Conv3d(args.num_channels, args.num_channels * 2, 3, stride=1, padding=1)
self.conv3 = nn.Conv3d(args.num_channels * 2, args.num_channels * 2, 3, stride=1)
self.conv4 = nn.Conv3d(args.num_channels * 2, args.num_channels * 2, 3, stride=1)
self.conv5 = nn.Conv3d(args.num_channels * 2, args.num_channels, 1, stride=1)

self.bn1 = nn.BatchNorm3d(args.num_channels)
self.bn2 = nn.BatchNorm3d(args.num_channels * 2)
self.bn3 = nn.BatchNorm3d(args.num_channels * 2)
self.bn4 = nn.BatchNorm3d(args.num_channels * 2)
self.bn5 = nn.BatchNorm3d(args.num_channels)

self.fc1 = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4)*(self.board_z-4), 1024) #4096 -> 1024
self.fc_bn1 = nn.BatchNorm1d(1024)

self.fc2 = nn.Linear(1024, 512)
self.fc_bn2 = nn.BatchNorm1d(512)

self.fc3 = nn.Linear(512, 512)
self.fc_bn3 = nn.BatchNorm1d(512)

# output p(st)
self.fc4 = nn.Linear(512, self.action_size)

# output vt
self.fc5 = nn.Linear(512, 1)
```

## Monte-Carlo Tree Search for Policy Improvement
Given a state **s**, the neural network provides an estimate of the policy **pθ**

## Feature engineering(ONLY USED IN SUPERVISED LEARNING WHICH IS DEPRECATED)
This was by far the most annoying process in the building of the neural network architecture, from the board serialization data to type of data...

### Board Serialization
My approach to the board serializaton/encoding was to make it have every feature it could need about the board while not making it too complex, mainly because i didn't have the GPU or disk hardware to process complex datasets, the board was encoded in different ways, having a final shape of <b>(24, 8, 8)</b>

| Feature | Description | Shape |
| --- | --- | --- |
| Piece Bitmaps | Bitmaps of pices from the different colors, each piece for each color(6*2=12) | (12, 8, 8) |
| Turn | Binary feature to indicate who is playing next | (1, 8, 8) |
| Material | Integer value for the chess piece relative value count for each player | (2, 8, 8) |
| Material Difference | Material difference(`White.Material - Black.Material`) | (1, 8, 8) |
| En-Passant Square | Integer value for the en-passant square from 1-64, 0 if None | (1, 8, 8) |
| Attacking Squares | Bitmap of attacking squares for each color | (2, 8, 8) |
| Castling Rights | Binary features for each castling right option(kingside and queenside) for each player | (4, 8, 8) |
| PLY | Ply move count(half-moves) | (1, 8, 8)

### Float32 vs Int8
Choosing the datatype of the board serialization values played a big part in saving RAM, GPU computations and disk space.
A 2.5M dataset in `float32` would take 15GiB of RAM and disk space.
While a 5M dataset in `uint8` would take around 5GiB of RAM and disk space.

(I only had 16GiB to work with)

`float32` also brought problems such as the network not understanding the pieces bitmaps, because they were float values the network was thinking of the pieces as raw values instead of classes.

