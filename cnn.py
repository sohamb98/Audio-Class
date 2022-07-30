import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from ast import literal_eval
import re

def str_np(data_string):
    nparr = re.sub('(\d) +(-|\d)', r'\1,\2', data_string)
    nparr = nparr.replace(" ","")
    nparr = nparr.replace('\n',',')
    nparr = literal_eval(nparr)
    return nparr

currpath = os.path.abspath(os.getcwd())
feature_path = currpath + "/features3.csv"
feature_path = os.path.abspath(feature_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters

num_epochs = 90
batch_size = 128
learning_rate = 0.001



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
    'Tram':9
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
    9:'Tram'
}

for epoch in range(num_epochs):
    for chunk in pd.read_csv(feature_path, chunksize=batch_size,sep=";",error_bad_lines=False):
        vals = []
        labels = []
        
        for index_num,row in chunk.iterrows():
            #Adding extra dimension [512,431]  to  [1, 512,431] and then appending to get [n, 1, 512,431]
            vals.append([str_np(row["feature"])])
            labels.append(label_to_idx [ row["class"] ])
            spectograms = torch.from_numpy (np.array(vals))
            
        #print(np.shape(vals))
        tlabels = torch.from_numpy(np.array(labels))
        spectograms = torch.from_numpy (np.array(vals))
        #print( list(spectograms.size()) )
        tlabels = torch.from_numpy(np.array(labels))
        
        spectograms = spectograms.to(device)
        #Converting double tensor to float tensor as our model is float
        spectograms = spectograms.float()
        tlabels = tlabels.to(device)

        #Forward pass
        outputs = model(spectograms)
        #print( list(outputs.size()) )
        #print( list(tlabels.size()) )
        loss = criterion(outputs, tlabels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Done")
    print(f"Completed Epoch {epoch}/{num_epochs}")
    

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)
