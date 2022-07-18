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


currpath = os.path.abspath(os.getcwd())
feature_path = currpath + "/features.csv"
feature_path = os.path.abspath(feature_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#chunksize=1000000
#nrows=100
feature_chunk = pd.read_csv(feature_path, nrows=100 )
feature_data = feature_chunk
#pd_df = pd.concat(feature_chunk)
print(feature_data.head())

abc = feature_data.iat[0, feature_data.columns.get_loc('feature')]
abc = abc.replace('\n',',')
s2 = re.sub('(\d) +(-|\d)', r'\1,\2', abc)

#print(s2)
temp_np = np.array(literal_eval(s2))
print(np.shape(temp_np))
print(temp_np.min())
print(temp_np.max())
#print(abc)
#
#abc = abc.replace('\n', ',')

#temp_np = np.array(literal_eval(abc))

#print(abc[1:200])

#spectogram = np.array([])
#spectogram = feature_data['feature'].values
#print(spectogram)
#test = feature_data[['feature']].to_numpy()
#print(test)

#records = len(feature_data.index)
#for index_num,row in feature.iterrows():
#    row["feature"]
#    row["class"]