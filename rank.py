#import torch

#PATH = "./cnnDCASE.pth.tar"
#checkpoint = torch.load(PATH)

#print( checkpoint['state_dict'].keys())
#print(checkpoint['optimizer'].keys())
#print( checkpoint['state_dict']['conv2_1.weight'].shape )

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle

num_epochs = 200
batch_size = 32
learning_rate = 0.001

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


class AudioTestDataset(Dataset):
    def __init__(self):
        #data loading
        currpath = os.path.abspath(os.getcwd())
        feature_path = currpath + "/DCASEevaluate_features1.pickle"
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

dataset = AudioTestDataset()
test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = "./cnnDCASE.pth.tar"




class DCASENet(nn.Module):
    def __init__(self):
        super(DCASENet, self).__init__()
        # n, 1, 40, 51 
        self.conv1_1 = nn.Conv2d(1, 16, 7, 1, 3)  # n, 16, 40, 51 
        self.Batch1_1 = nn.BatchNorm2d(16)  # n, 16, 40, 51 



        self.conv2_1 = nn.Conv2d(16, 16, 7, 1, 3)
        self.Batch2_1 = nn.BatchNorm2d(16)  # -> n, 16, 40, 51
        self.pool2 = nn.MaxPool2d(5,5)       
        self.dropout2 = nn.Dropout(p=0.3)  # -> n, 16,  8, 10 


        

        self.conv3_1 = nn.Conv2d(16, 32, 7, 1, 3)  
        self.Batch3_1 = nn.BatchNorm2d(32) # -> n, 32,  8, 10 

        self.pool3 = nn.MaxPool2d(4,stride=(4,100))    
        self.dropout3 = nn.Dropout(p=0.3)  # -> n, 32,  2, 1 

        
        #Flatten to n, 512, 28, 23


        self.fc1 = nn.Linear(32*2*1 , 100)
        self.dropout4 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(100, 10)
        
        
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        #print(x.shape)
        # -> n, 1, 40, 51
        x = self.conv1_1(x)
        x = self.Batch1_1(x)
        x = F.relu(x)

        #print(x.shape)
        # -> n, 16, 40, 51

        #Layer 2
        x = self.conv2_1(x)
        x = self.Batch2_1(x)
        x = F.relu(x)
        #print(x.shape)
        # -> n, 32, 40, 51
        x = self.pool2(x)
        x = self.dropout2(x)

        #print(x.shape)
        # -> n, 32,  8, 10

        #Layer 3
        x = self.conv3_1(x)
        x = self.Batch3_1(x)
        x = F.relu(x)
        # -> n, 32,  8, 10
        #print(x.shape)
        x = self.pool3(x)
        x = self.dropout3(x)
        # -> n, 64,  2, 1
        #print(x.shape)


        x = x.view(-1, 32*2*1) #Flatten

        x = F.relu(self.fc1(x)) # Dense  
        x = self.dropout4(x) 

                                #Output Layer
        x = self.fc2(x)
        x = self.logSoftmax(x)

        return x


model = DCASENet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer =  torch.optim.Adam(model.parameters(), lr=learning_rate)

checkpoint = torch.load(PATH)

model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])


from torch import linalg as LA
lname ="conv1_1.weight"
holder = 0
flag = 0
dimens = []
n_total_steps = len(test_loader)
for i, (spectograms, labels) in enumerate(test_loader):
    spectograms = spectograms.float()
    spectograms = spectograms.to(device)
    labels = labels.to(device)
    outputs = model(spectograms)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    for name, param in model.named_parameters():
        #print(param.weight.shape)
        print(name)
        print(param.data.shape)
        print(param.grad.shape)
        if name == lname:
            if flag == 0:
                dimens = param.data.shape
                holder = torch.zeros(dimens[0],dimens[1],dimens[2],dimens[3],device=device)
                #holder = torch.zeros(16,1,7,7,device=device)
                flag = 1    
            holder = holder.add(torch.abs(torch.mul(param.data, param.grad)))


    print("Done!")
    optimizer.step()
    print(f"Step: {i+1}/{n_total_steps}")

'''
Filter Wise Normalization
#print(holder)
ranks = holder.view(16,1*7*7)
#print(ranks)
l2norm = LA.vector_norm(ranks, ord=2,dim=[1] )
print(ranks.shape)
print(l2norm.shape)
divider = l2norm.reshape((-1,1))
print(divider.shape)
normranks = torch.div(ranks,divider)
print(normranks)
'''
#layerWise
#ranks = holder.view(16,1*7*7)
ranks = holder.view(dimens[0],dimens[1]*dimens[2]*dimens[3])
l2norm = LA.vector_norm(ranks, ord=2)
normranks = torch.div(ranks,l2norm)
print(normranks.shape)
#print(model.conv1_1.weight.shape)