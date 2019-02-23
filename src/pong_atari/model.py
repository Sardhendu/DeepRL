
import torch

from torch import nn

import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, net_name='net1'):
        """
        
            input = tranjectories [horizon, num_channels, img_h, img_w]
                    num_channels = 2
                    img_h = 80
                    img_w = 80
                    
            output = actions probabilities [horizon, 1]
            
        :param net_name:    Name of the Neural architecture to use
        
            The network will simply output a probability value, that would suggest the
        """
        super(Model, self).__init__()
        # 80x80x2 to 38x38x4
        # 2 channel from the stacked frame
        self.net_name = net_name
        
        if net_name == 'net1':
            self.net1()
            self.forward_func = self.forward_net1
            
        elif net_name == 'net2':
            self.net2()
            self.forward_func = self.forward_net2
            
        else:
            raise ValueError('Only net1 and net2 excepted ....')
        
    def net1(self):
        print ('[Model] Initializing net 1')
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=6, stride=2, bias=False)
        # 38x38x4 to 9x9x32
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=6, stride=4)
        self.size = 9 * 9 * 16
        
        # two fully connected layer
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)
        
        # Sigmoid to
        self.sig = nn.Sigmoid()
        
    def net2(self):
        """
        Note the images are not very complex, hence having a deep network could result in
        overfitting.

        :return:
        """
        print('[Model] Initializing net 2')
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=7, stride=2, bias=False)
        # 38x38x4 to 9x9x32
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=4, stride=4)
    
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.size = 7 * 7 * 32
    
        # two fully connected layer
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)
    
        # Sigmoid to
        self.sig = nn.Sigmoid()
    
    def forward_net1(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        return self.sig(self.fc2(x))
    
    def forward_net2(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        return self.sig(self.fc2(x))
    
    def forward(self, x):
        return self.forward_func(x)
    
    