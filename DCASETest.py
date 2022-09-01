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
#PATH = "./cnnDCASE.pth.tar"
PATH = "./cnnDCASEprun1retrain.pth.tar"




class DCASENet(nn.Module):
    def __init__(self):
        super(DCASENet, self).__init__()
        # n, 1, 40, 51 
        self.conv1_1 = nn.Conv2d(1, 12, 7, 1, 3)  # n, 16, 40, 51 
        self.Batch1_1 = nn.BatchNorm2d(12)  # n, 16, 40, 51 



        self.conv2_1 = nn.Conv2d(12, 16, 7, 1, 3)
        self.Batch2_1 = nn.BatchNorm2d(16)  # -> n, 16, 40, 51
        self.pool2 = nn.MaxPool2d(5,5)       
        self.dropout2 = nn.Dropout(p=0.3)  # -> n, 16,  8, 10 


        

        self.conv3_1 = nn.Conv2d(16, 30, 7, 1, 3)  
        self.Batch3_1 = nn.BatchNorm2d(30) # -> n, 32,  8, 10 

        self.pool3 = nn.MaxPool2d(4,stride=(4,100))    
        self.dropout3 = nn.Dropout(p=0.3)  # -> n, 32,  2, 1 

        
        #Flatten to n, 512, 28, 23


        self.fc1 = nn.Linear(30*2*1 , 100)
        self.dropout4 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(100, 10)
        
        
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        print(x.shape)
        # -> n, 1, 40, 51
        x = self.conv1_1(x)
        x = self.Batch1_1(x)
        x = F.relu(x)

        print(x.shape)
        # -> n, 16, 40, 51

        #Layer 2
        x = self.conv2_1(x)
        x = self.Batch2_1(x)
        x = F.relu(x)
        print(x.shape)
        # -> n, 32, 40, 51
        x = self.pool2(x)
        x = self.dropout2(x)

        print(x.shape)
        # -> n, 32,  8, 10

        #Layer 3
        x = self.conv3_1(x)
        x = self.Batch3_1(x)
        x = F.relu(x)
        # -> n, 32,  8, 10
        print(x.shape)
        x = self.pool3(x)
        x = self.dropout3(x)
        # -> n, 64,  2, 1
        print(x.shape)


        x = x.view(-1, 30*2*1) #Flatten

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
model.eval()

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        print(labels.shape)
        outputs = model(images)
        

        _, predicted = torch.max(outputs, 1)

        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(outputs.shape[0]):
             label = labels[i]
             pred = predicted[i]
             if (label == pred):
                n_class_correct[label] += 1
             n_class_samples[label] += 1


        #print(predicted)

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {idx_to_label[i]}: {acc} %')