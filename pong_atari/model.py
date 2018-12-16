

from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        
        
        ########
        ##
        ## Modify your neural network
        ##
        ########
        
        # 80x80 to outputsize x outputsize
        # outputsize = (inputsize - kernel_size + stride)/stride
        # (round up if not an integer)
        
        # output = 20x20 here
        self.conv = nn.Conv2d(2, 1, kernel_size=4, stride=4)
        self. size = 1 * 20 *20
        
        # 1 fully connected layer
        self.fc = nn.Linear(self.size, 1)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        
        ########
        ##
        ## Modify your neural network
        ##
        ########
        
        x = F.relu(self.conv(x))
        # flatten the tensor
        x = x.view(-1 ,self.size)
        return self.sig(self.fc(x))
