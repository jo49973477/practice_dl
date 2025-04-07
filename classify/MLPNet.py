import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.LazyLinear(2048)
        self.layer2 = nn.Linear(2048, 2048)
        self.layer3 = nn.Linear(2048, 10)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        hidden1 = self.layer1(x)
        hidden1 = self.activation(hidden1)
        
        hidden2 = self.layer2(hidden1)
        hidden2 = self.activation(hidden2)
        
        hidden3 = self.layer3(hidden2)
        output = self.activation(hidden3)
        
        return output
    
if __name__ == "__main__":
    x1 = torch.randn(4, 1024*3)
    x2 = torch.randn(16, 1024*3)
    x3 = torch.randn(64, 1024*3)
    
    model = MLP()
    print(model(x1).size())
    print(model(x2).size())
    print(model(x3).size())