import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
        ones = int(self.data[:, 6].sum())
        print(ones)
        i = 0
        while self.data.shape[0] > 12000:
            if self.data[i][6] == 0:   
                self.data = np.delete(self.data, i, 0)
            i += 1
            if i >= self.data.shape[0]:
                i = 0
        print(self.data.shape)
        print(self.data)
        
        ones = int(self.data[:, 6].sum())
        print(ones)
        
        # STUDENTS: it may be helpful for the final part to balance the distribution of your collected data
        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
        # STUDENTS: __len__() returns the length of the dataset
        return len(self.normalized_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        # STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
        # x and y should both be of type float32. There are many other ways to do this, but to work with autograding
        # please do not deviate from these specifications.
        x = np.float32(self.normalized_data[idx, :6])
        y = np.float32(self.normalized_data[idx, 6])
        my_dict = {'input' : x, 'label' : y}
        return my_dict

class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        # STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader    
        # make sure your split can handle an arbitrary number of samples in the dataset as this may vary
        train_data_len = int(0.8*self.nav_dataset.__len__())
        test_data_len = int(0.2*self.nav_dataset.__len__())
        train_data, test_data = data.random_split(self.nav_dataset, [train_data_len, test_data_len])
        
        self.train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
        
        
def main():
    batch_size = 1000
    data_loaders = Data_Loaders(batch_size)
    print(data_loaders.test_loader)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
