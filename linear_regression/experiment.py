from dataset import LinearRegressionDataset
import numpy as np
import torch
import torch.nn as nn

dataset = LinearRegressionDataset(train= True)
data = dataset[69]
x, y = torch.Tensor(data["x"]), data["y"]
model = nn.Linear(4,3)
y_pred = model(x)
print("prediction:", y_pred)
print("answer:", y)