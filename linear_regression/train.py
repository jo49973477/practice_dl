from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LinearRegressionDataset

EPOCH = 20
LEARNING_RATE = 1e-2
BATCH_SIZE = 64

train_dataset = LinearRegressionDataset(train= True)
test_dataset = LinearRegressionDataset(train= False)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = model = nn.Linear(4,3)
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

avg_train_losses = []
avg_test_losses = []
for epoch in range(1, EPOCH + 1):
    loss_val = 60
    
    #train
    batch_losses = []
    model.train()
    for i, data in tqdm(enumerate(train_dataloader), 
                     desc = "epoch {} training...".format(epoch)):
        label, x = data["label"], data["input"]
        pred = model(x)
        
        loss = loss_function(pred, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        batch_losses.append(loss.item())
        avg_train_losses.append(sum(batch_losses) /(i+1))
    
    #test
    model.eval()
    avg_test_loss = 0
    for data in tqdm(test_dataloader, 
                     desc = "epoch {} testing...".format(epoch, loss_val)):
        label, x = data["label"], data["input"]
        pred = model(x)
        
        loss = loss_function(pred, label)
        avg_test_loss += loss.item() / len(test_dataloader)
        
    avg_test_losses.append(avg_test_loss)
        
    print("EPOCH {} TEST LOSS: {}".format(epoch, avg_test_loss))
    
plt.plot(np.linspace(0, EPOCH, EPOCH*len(train_dataloader)), avg_train_losses, label="train")
plt.plot(list(range(1, EPOCH+1)), avg_test_losses, label="test")
plt.title("Train and test loss")
plt.savefig("plot.png") 