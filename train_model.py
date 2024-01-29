from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF
import numpy as np
import pickle

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle

def train_model(no_epochs):

    batch_size = 1000
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()
    
    learning_rate = 0.01
    momentum = 0.5
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    loss_function = nn.MSELoss()
    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)


    for epoch_i in range(no_epochs):
        model.train()
        for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
            inp = sample['input']
            lab = sample['label']
            optimizer.zero_grad()
            output = model(inp)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
        min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        losses.append(min_loss)

    pickle.dump(model, open("saved/saved_model.pkl", "wb"))


if __name__ == '__main__':
    no_epochs = 10
    train_model(no_epochs)
