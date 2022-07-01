from email.mime import audio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

from glob import glob

import librosa
import librosa.display

from playsound import playsound

#Reading metadata
metadata = pd.read_csv(r"D:\Audio Dataset\TAU-urban-acoustic-scenes-2020-mobile-development\meta.csv", delim_whitespace=True)
metadata.head()
#print(metadata['filename'][0])

#Redeading metadata from the folder
audio_files = glob("D:\\Audio Dataset\\TAU-urban-acoustic-scenes-2020-mobile-development\\audio\\*.wav")


'''''
#print(audio_files[1])
playsound(audio_files[0])

y, sr = librosa.load(audio_files[0])

print(f'y:{y[:10]}')
print(f'shape y:{y.shape}')
print(f'sr :{sr}')

pd.Series(y).plot(figsize=(10,5))

plt.show()

#Creating spectogram

s = librosa.stft(y)

#converting to decibles from aplitude
s_db = librosa.amplitude_to_db(np.abs(s), ref=np.max)
print(s_db.shape)


fig, ax = plt.subplots(figsize=(10,5))
img = librosa.display.specshow(s_db, x_axis="time", y_axis="log")
fig.colorbar(img, ax=ax, format=f'%0.2f')
plt.show()




# Creating log mel spectogram
s1 = librosa.feature.melspectrogram(y=y,sr = sr, n_mels=128*4)
#Converting to db
s1_db_mel = librosa.amplitude_to_db(np.abs(s1), ref=np.max)

fig, ax = plt.subplots(figsize=(15,5))
img = librosa.display.specshow(s1_db_mel, x_axis="time", y_axis="log")
fig.colorbar(img, ax=ax, format=f'%0.2f')
plt.show()
'''''

def features_extracter(file):
    y, sr = librosa.load(file)
    s1 = librosa.feature.melspectrogram(y=y,sr = sr, n_mels=128*4)
    s1_db_mel = librosa.amplitude_to_db(np.abs(s1), ref=np.max)

    return s1_db_mel

def main():
    features = []
    records = len(metadata.index)
    for index_num,row in metadata.iterrows():
        filename = os.path.abspath( row['filename'] )
        final_class_labels = row['scene_label']
        feature = features_extracter(filename)
        features.append([feature, final_class_labels])
        os.system('cls')
        print(f'Extracted file{index_num}/{records}')


    
    #print(os.path.abspath( metadata['filename'][0] ))
    #print(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    main()






