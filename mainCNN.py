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
        #feature_path = currpath + "/train_features.pickle"
        feature_path = currpath + "/DCASEtrain_features1.pickle"
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



device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

#hyper parameters
dataset = AudioDataset()


num_epochs = 200
batch_size = 16
learning_rate = 0.005

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

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        #in_channels, out_channels, kernel_size
        # n, 1, 512, 431 
        self.conv1_1 = nn.Conv2d(1, 64, 3,padding=1)  # n, 64, 40, 51 
        self.conv1_2 = nn.Conv2d(64, 64, 3,padding=1) # n, 64, 40, 51 
        self.Batch1_1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2,2)    # n, 64, 20, 25
        self.dropout1 = nn.Dropout(p=0.3)

        self.conv2_1 = nn.Conv2d(64, 128, 3,padding=1)  # n, 128, 20, 25 
        self.conv2_2 = nn.Conv2d(128, 128, 3,padding=1) # n, 128, 20, 25 
        self.Batch2_1 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2,2)    # n, 128, 10, 12 
        self.dropout2 = nn.Dropout(p=0.3)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)  # n, 256, 10, 12 
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1) # n, 256, 10, 12
        self.Batch3_1 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2,2)    # n, 256, 5, 6
        self.dropout3 = nn.Dropout(p=0.3)

        self.conv4_1 = nn.Conv2d(256, 512, 3,padding=1)  # n, 512, 5, 6 
        self.conv4_2 = nn.Conv2d(512, 512, 3,padding=1) # n, 512, 5, 6 
        self.Batch4_1 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(4,stride=(4,100))    # n, 512, 1, 1
        self.dropout4 = nn.Dropout(p=0.3)
        #Flatten to n, 512, 28, 23


        self.fc1 = nn.Linear(512*1*1, 100)
        self.fc2 = nn.Linear(100, 10)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # -> n, 1, 512, 431
        #print(x.shape)
        x = self.conv1_1(x)
        #print(x.shape)
        x = self.conv1_2(x)
        #print(x.shape)
        x = self.Batch1_1(x)
        x = self.dropout1(self.pool1(F.relu(x)))
        #print(x.shape)

        x = self.conv2_1(x)
        #print(x.shape)
        x = self.conv2_2(x)
        #print(x.shape)
        x = self.Batch2_1(x)
        x = self.dropout2(self.pool2(F.relu(x)))


        #print(x.shape)
        x = self.conv3_1(x)
        #print(x.shape)
        x = self.conv3_2(x)
        #print(x.shape)
        x = self.Batch3_1(x)
        x = self.dropout2(self.pool3(F.relu(x)))
        


        #print(x.shape)
        x = self.conv4_1(x)
        #print(x.shape)
        x = self.conv4_2(x)
        #print(x.shape)
        x = self.Batch4_1(x)
        x = self.dropout4(self.pool4(F.relu(x)))
        #print(x.shape)

        x = x.view(-1, 512*1*1)

        x = F.relu(self.fc1(x))               
        x = self.fc2(x) 
 

        x = self.logSoftmax(x)

        return x


model = VGGNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

lossar  = []
epochar = [] 

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
    lossar.append(epoch)
    epochar.append(loss.item())


plt.plot(lossar,epochar)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Training Loss vs Epoch")
plt.savefig('VGGTrainingLoss.png')    

print('Finished Training')
PATH = './cnnVGG.pth.tar'
torch.save( { 'epoch': epoch,'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(), 'loss': criterion}, PATH)