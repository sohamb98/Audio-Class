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
lname1 ="conv2_1.weight"
lname2 = "conv3_1.weight"
holder = 0
holder1 = 0
holder2 = 0

flag = 0
flag1 = 0
flag2 = 0

dimens = []
dimens1 = []
dimens2 = []
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
        #print(name)
        print(param.data.shape)
        #print(param.grad.shape)
        if name == lname:
            if flag == 0:
                dimens = param.data.shape
                holder = torch.zeros(dimens[0],dimens[1],dimens[2],dimens[3],device=device)
                #holder = torch.zeros(16,1,7,7,device=device)
                flag = 1    
            holder = holder.add(torch.abs(torch.mul(param.data, param.grad)))
        if name == lname1:
            if flag1 == 0:
                dimens1 = param.data.shape
                holder1 = torch.zeros(dimens1[0],dimens1[1],dimens1[2],dimens1[3],device=device)
                #holder = torch.zeros(16,16,7,7,device=device)
                flag1 = 1    
            holder1 = holder1.add(torch.abs(torch.mul(param.data, param.grad)))
        if name == lname2:
            if flag2 == 0:
                dimens2 = param.data.shape
                holder2 = torch.zeros(dimens2[0],dimens2[1],dimens2[2],dimens2[3],device=device)
                #holder = torch.zeros(32,16,7,7,device=device)
                flag2 = 1    
            holder2 = holder2.add(torch.abs(torch.mul(param.data, param.grad)))
            #print(holder2.shape)

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

#layer0
#ranks = holder.view(16,1*7*7)
ranks = holder.view(dimens[0],dimens[1]*dimens[2]*dimens[3])
l2norm = LA.vector_norm(ranks, ord=2)
normranks = torch.div(ranks,l2norm)
print(normranks.shape)
#print(model.conv1_1.weight.shape)

#layer1
ranks1 = holder1.view(dimens1[0],dimens1[1]*dimens1[2]*dimens1[3])
l2norm1 = LA.vector_norm(ranks1, ord=2)
normranks1 = torch.div(ranks1,l2norm1)
print(normranks1.shape)

#layer2
ranks2 = holder2.view(dimens2[0],dimens2[1]*dimens2[2]*dimens2[3])
l2norm2 = LA.vector_norm(ranks2, ord=2)
normranks2 = torch.div(ranks2,l2norm2)
print(normranks2.shape)


npnormranks = normranks.cpu().numpy()
npnormranks1 = normranks1.cpu().numpy()
npnormranks2 = normranks2.cpu().numpy()
unodict_ranks = {}

for i in range(dimens[0]):
    sum = 0
    for j in range(dimens[1]*dimens[2]*dimens[3]):
        sum = sum + (npnormranks[i][j]*npnormranks[i][j])
    unodict_ranks[sum] = (i,0)

for i in range(dimens1[0]):
    sum = 0
    for j in range(dimens1[1]*dimens1[2]*dimens1[3]):
        sum = sum + (npnormranks1[i][j]*npnormranks1[i][j])
    unodict_ranks[sum] = (i,1)

for i in range(dimens2[0]):
    sum = 0
    for j in range(dimens2[1]*dimens2[2]*dimens2[3]):
        sum = sum + (npnormranks2[i][j]*npnormranks2[i][j])
    unodict_ranks[sum] = (i,2)

odict_ranks = collections.OrderedDict(sorted(unodict_ranks.items()))
nfilters = len(odict_ranks)
percentage = 0.05
k = int(percentage*nfilters)

prunelist = {0:[],1:[],2:[]}

i=0
for (key, value),i in zip (odict_ranks.items() , range(k)):
    print(key, value)
    prunelist[value[1]].append(value[0])

for i in range(3):
    prunelist[i].sort(reverse=True)
print(prunelist)

print( checkpoint['state_dict'].keys())

paras = ['conv1_1.weight','conv1_1.bias','Batch1_1.weight','Batch1_1.bias','Batch1_1.running_mean','Batch1_1.running_var']
paras1 = ['conv2_1.weight','conv2_1.bias','Batch2_1.weight','Batch2_1.bias','Batch2_1.running_mean','Batch2_1.running_var']
paras2 = ['conv3_1.weight','conv3_1.bias','Batch3_1.weight','Batch3_1.bias','Batch3_1.running_mean','Batch3_1.running_var']
fcparas = ['fc1.weight','fc1.bias','fc2.weight','fc2.weight']

checkpoint = torch.load(PATH)


def prunelayer(params, row_exclude):
    for param in params:
        for r in row_exclude:
            x = checkpoint['state_dict'][param]
            x = torch.cat((x[:r],x[r+1:]))
            checkpoint['state_dict'][param] = x
            #print(checkpoint['state_dict'][param].shape)

def prunenextlayer(nextlayer,row_exclude):
    for r in row_exclude:
        x = checkpoint['state_dict'][nextlayer] 
        #print(x.shape)
        x = torch.cat((x[:,:r],x[:,r+1:]),dim=1)
        checkpoint['state_dict'][nextlayer] = x
        #print(checkpoint['state_dict'][nextlayer].shape)

#pruning Layer 1
prunelayer(paras, prunelist[0])
prunenextlayer(paras1[0],prunelist[0])
#Pruning Layer 2
prunelayer(paras1, prunelist[1])
prunenextlayer(paras2[0],prunelist[1])
#Pruning Layer 3
prunelayer(paras2, prunelist[2])

#prunenextfclayer
x = checkpoint['state_dict'][fcparas[0]]
p=32 # Original No of filters in layer
x = torch.reshape(x,(-1,p,2,1))

for r in prunelist[2]:
    #print(x.shape)
    x = torch.cat((x[:,:r],x[:,r+1:]),dim=1)
    #print(checkpoint['state_dict'][nextlayer].shape)

x = torch.reshape(x,(-1,(p-len(prunelist[2]))*2*1))
checkpoint['state_dict'][fcparas[0]] = x

print(checkpoint['state_dict'][paras[0]].shape)
print(checkpoint['state_dict'][paras1[0]].shape)
print(checkpoint['state_dict'][paras2[0]].shape)
print(checkpoint['state_dict'][fcparas[0]].shape)
print(checkpoint['state_dict'][fcparas[1]].shape)

#Next Modify the optimizer:
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

affected_params = {0 : [0,1,2,3], 1:[4,5,6,7], 2: [8,9,10,11], 3:[12,13] } 

for i in range(3):
    for para in affected_params[i]:
        x = checkpoint['optimizer']['state'][para]['exp_avg']
        y = checkpoint['optimizer']['state'][para]['exp_avg_sq']
        for row_exclude in prunelist[i]:
            x = torch.cat((x[:row_exclude],x[row_exclude+1:]))
            y = torch.cat((y[:row_exclude],y[row_exclude+1:]))
        checkpoint['optimizer']['state'][para]['exp_avg'] = x
        checkpoint['optimizer']['state'][para]['exp_avg_sq'] = y
        

for i,j in zip([4,8], [0,1]):
    #print(i,j)
    x = checkpoint['optimizer']['state'][i]['exp_avg']
    y = checkpoint['optimizer']['state'][i]['exp_avg_sq'] 
    
    for row_exclude in prunelist[j]:
        x = torch.cat((x[:,:row_exclude],x[:,row_exclude+1:]),dim=1)
        y = torch.cat((y[:,:row_exclude],y[:,row_exclude+1:]),dim=1)
    checkpoint['optimizer']['state'][i]['exp_avg'] = x
    checkpoint['optimizer']['state'][i]['exp_avg_sq'] = y
    
print(checkpoint['optimizer']['state'][0]['exp_avg'].shape)
print(checkpoint['optimizer']['state'][4]['exp_avg'].shape)
print(checkpoint['optimizer']['state'][8]['exp_avg'].shape)


x = checkpoint['optimizer']['state'][12]['exp_avg']
y = checkpoint['optimizer']['state'][12]['exp_avg_sq']
x = torch.reshape(x,(-1,p,2,1))
y = torch.reshape(y,(-1,p,2,1))
for r in prunelist[2]:
    #print(x.shape)
    x = torch.cat((x[:,:r],x[:,r+1:]),dim=1)
    y = torch.cat((y[:,:r],y[:,r+1:]),dim=1)
    #print(checkpoint['state_dict'][nextlayer].shape)

x = torch.reshape(x,(-1,(p-len(prunelist[2]))*2*1))
y = torch.reshape(x,(-1,(p-len(prunelist[2]))*2*1))
checkpoint['optimizer']['state'][12]['exp_avg'] = x
checkpoint['optimizer']['state'][12]['exp_avg_sq'] = y
print(checkpoint['optimizer']['state'][12]['exp_avg_sq'].shape)
'''
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
'''
PATH = "./cnnDCASEprun1.pth.tar"
torch.save( { 'epoch': checkpoint['epoch'],'state_dict': checkpoint['state_dict'], 'optimizer' : checkpoint['optimizer'], 'loss': checkpoint['loss']}, PATH)
