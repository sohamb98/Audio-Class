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
    data_string = data_string.replace('\n',',')
    data_string = re.sub('(\d) +(-|\d)', r'\1,\2', data_string)
    #nparr = np.array(literal_eval(data_string))
    nparr = literal_eval(data_string)
    return nparr

currpath = os.path.abspath(os.getcwd())
feature_path = currpath + "/features.csv"
feature_path = os.path.abspath(feature_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters

num_epochs = 1
batch_size = 5
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
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)            # -> n, 400
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
    for chunk in pd.read_csv(feature_path, chunksize=batch_size):
        vals = []
        labels = []
        #vals = np.empty((512,431), float)
        #labels = np.empty((0,413), int)
        for index_num,row in chunk.iterrows():
            abc = str_np(row["feature"])
            vals.append(abc)
            labels.append(label_to_idx [ row["class"] ])
            spectograms = torch.from_numpy (np.array(vals))
            tlabels = torch.from_numpy(np.array(labels))
            #print(np.shape(abc))
            #vals = np.append(vals, str_np(row["feature"]), axis=1)
            #label = np.append(labels, label_to_idx [ row["label"] ], axis = 1)
        
        
        spectograms = spectograms.to(device)
        tlabels = tlabels.to(device)

        #Forward pass
        outputs = model(spectograms)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Done")

print('Finished Training')
