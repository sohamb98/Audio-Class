from email.mime import audio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob

import librosa
import librosa.display

from playsound import playsound

audio_files = glob("D:\\Audio Dataset\\TAU-urban-acoustic-scenes-2020-mobile-development\\audio\\*.wav")
#print(audio_files[1])
playsound(audio_files[0])

y, sr = librosa.load(audio_files[0])

print(f'y:{y[:10]}')
print(f'shape y:{y.shape}')
print(f'sr :{sr}')

pd.Series(y).plot(figsize=(10,5))

plt.show()