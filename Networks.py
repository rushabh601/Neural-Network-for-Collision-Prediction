import torch
import torch.nn as nn
from Data_Loaders import Data_Loaders

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        # STUDENTS: __init__() must initiatize nn.Module and define your network's
        # custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.input_to_hidden1 = nn.Linear(6, 12)
        self.input_to_hidden2 = nn.Linear(12, 12)
        self.nonlinear_actv = nn.ReLU()
        self.hidden_to_output = nn.Linear(12, 1)
        self.nonlinear_output = nn.Sigmoid()
        pass

    def forward(self, input1):
        # STUDENTS: forward() must complete a single forward pass through your network
        # and return the output which should be a tensor
        hidden = self.nonlinear_actv(self.input_to_hidden1(input1))
        hidden = self.nonlinear_actv(self.input_to_hidden2(hidden))
        print(hidden)
        hidden = self.hidden_to_output(hidden)
        output = self.nonlinear_output(hidden)
        return output


    def evaluate(self, model, test_loader, loss_function):
        # STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
        # mind that we do not need to keep track of any gradients while evaluating the
        # model. loss_function will be a PyTorch loss function which takes as argument the model's
        # output and the desired output.
        i = 0
        loss = 0
        for idx, sample in enumerate(test_loader):
            input1, label = sample['input'], sample['label']
            network_output = model.forward(input1)
            loss += loss_function(network_output, label)
            i += 1
        loss = loss/i
        
        return loss
        

def main():
    batch_size = 1000
    data_loader = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()
    loss_function = nn.MSELoss()
    loss = model.evaluate(model, data_loader.test_loader, loss_function)


if __name__ == '__main__':
    main()
