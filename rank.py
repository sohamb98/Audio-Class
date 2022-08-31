#import torch

#PATH = "./cnnDCASE.pth.tar"


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
import collections

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
npnormranks = normranks.cpu().numpy()
unodict_ranks = {}

for i in range(dimens[0]):
    sum = 0
    for j in range(dimens[1]*dimens[2]*dimens[3]):
        sum = sum + (npnormranks[i][j]*npnormranks[i][j])
    unodict_ranks[sum] = i

odict_ranks = collections.OrderedDict(sorted(unodict_ranks.items()))

K = 0
i=0
s = ()
for key, value in odict_ranks.items():
    if(K==i):
        s = (key,value)
        print(key,value)
        break
    i=i+1
#
#
#
#
#checkpoint = torch.load(PATH)

#print( checkpoint['state_dict'].keys())
#print(checkpoint['optimizer']['state'])
#print( checkpoint['state_dict']['conv2_1.weight'].shape )

paras = ['conv1_1.weight','conv1_1.bias','Batch1_1.weight','Batch1_1.bias','Batch1_1.running_mean','Batch1_1.running_var']
nextlayer = 'conv2_1.weight'

row_exclude = s[1]
for para in paras:
    x = checkpoint['state_dict'][para]
    x = torch.cat((x[:row_exclude],x[row_exclude+1:]))
    checkpoint['state_dict'][para] = x
    print(checkpoint['state_dict'][para].shape)


x = checkpoint['state_dict'][nextlayer] 
#print(x.shape)
x = torch.cat((x[:,:row_exclude],x[:,row_exclude+1:]),dim=1)
checkpoint['state_dict'][nextlayer] = x

#Pruning Optimizer
#0conv1_1.weight
#1conv1_1.bias
#2Batch1_1.weight
#3Batch1_1.bias
#4conv2_1.weight
#5conv2_1.bias
#6Batch2_1.weight
#7Batch2_1.bias
#8conv3_1.weight
#9conv3_1.bias
#10Batch3_1.weight
#11Batch3_1.bias
#12fc1.weight
#13fc1.bias
#14fc2.weight
#15fc2.bias
affected_params = [0,1,2,3] 
affected_params_nl = 4

for para in affected_params:
    x = checkpoint['optimizer']['state'][para]['exp_avg']
    y = checkpoint['optimizer']['state'][para]['exp_avg_sq']
    x = torch.cat((x[:row_exclude],x[row_exclude+1:]))
    y = torch.cat((y[:row_exclude],y[row_exclude+1:]))
    checkpoint['optimizer']['state'][para]['exp_avg'] = x
    checkpoint['optimizer']['state'][para]['exp_avg_sq'] = y
    print(checkpoint['optimizer']['state'][para]['exp_avg'].shape)

x = checkpoint['optimizer']['state'][affected_params_nl]['exp_avg']
y = checkpoint['optimizer']['state'][affected_params_nl]['exp_avg_sq'] 
x = torch.cat((x[:,:row_exclude],x[:,row_exclude+1:]),dim=1)
y = torch.cat((y[:,:row_exclude],y[:,row_exclude+1:]),dim=1)
checkpoint['optimizer']['state'][affected_params_nl]['exp_avg'] = x
checkpoint['optimizer']['state'][affected_params_nl]['exp_avg_sq'] = y
print(checkpoint['optimizer']['state'][affected_params_nl]['exp_avg'].shape)

PATH = "./cnnDCASEprun1.pth.tar"
print(x.shape)
torch.save( { 'epoch': checkpoint['epoch'],'state_dict': checkpoint['state_dict'], 'optimizer' : checkpoint['optimizer'], 'loss': checkpoint['loss']}, PATH)
