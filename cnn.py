import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from ast import literal_eval
import re
import pickle

#Dataloader Class

label_to_idx = {
    'airport':0,
    'bus':1,
    'metro':2,
    'metro_station':3,
    'park':4,
    'public_square':5,
    'shopping_mall':6,
    'street_pedestrian':7,
    'street_traffic':8,
    'tram':9
}

idx_to_label = {
    0:'airport',
    1:'bus',
    2:'metro',
    3:'metro_station',
    4:'park',
    5:'public_square',
    6:'shopping_mall',
    7:'street_pedestrian',
    8:'street_traffic',
    9:'tram'
}

class AudioDataset(Dataset):
    def __init__(self):
        #data loading
        currpath = os.path.abspath(os.getcwd())
        feature_path = currpath + "/train_features.pickle"
        feature_path = os.path.abspath(feature_path)

        with open(feature_path, "rb") as file:
            data = pickle.load(file)
        features = []
        #features = np.array(features)
        labels = []
        a = len(data)
        for i in range(a):
            #features = np.append( features, np.expand_dims(data[i][0], axis = 0)  )
            features.append([data[i][0]]) 
            labels.append(label_to_idx [ data[i][1] ]) 
        self.x = torch.from_numpy(np.array(features))
        self.y = torch.from_numpy(np.array(labels))
        self.n_samples = a
    def __getitem__(self,index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.n_samples



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters
dataset = AudioDataset()


num_epochs = 90
batch_size = 64
learning_rate = 0.001

train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)


#chunksize=1000000
#nrows=100
#feature_chunk = pd.read_csv(feature_path, nrows=100 )
#feature_data = feature_chunk
#pd_df = pd.concat(feature_chunk)
#print(feature_data.head())

#records = len(feature_data.index)

#abc = feature_data.iat[0, feature_data.columns.get_loc('feature')]
#abc = abc.replace('\n',',')
#s2 = re.sub('(\d) +(-|\d)', r'\1,\2', abc)
#print(s2)
#temp_np = np.array(literal_eval(s2))
#print(np.shape(temp_np))
#print(temp_np.min())
#print(temp_np.max())
#print(abc)
#

#for index_num,row in feature.iterrows():
#    row["feature"]
#    row["class"]

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 125 * 104, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 1, 512, 431
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 254, 213
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 125, 104
        x = x.view(-1, 16 * 125 * 104)            # -> n, 208000
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (spectograms, labels) in enumerate(train_loader):
        #Converting double tensor to float tensor as our model is float
        spectograms = spectograms.float()
        spectograms = spectograms.to(device)
        labels = labels.to(device)

        #Forward pass
        outputs = model(spectograms)
        #print( list(outputs.size()) )
        #print( list(tlabels.size()) )
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Step: {i+1}/{n_total_steps}, Epoch: {epoch}/{num_epochs}, loss: {loss.item():.4f}")
    print(f"Completed Epoch ")
    

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)
