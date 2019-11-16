import librosa
import soundfile as sf
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(X,sample_rate, mfcc, chroma, mel):
    if chroma:
        stft=np.abs(librosa.stft(X))
    result=np.array([])
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
    if chroma:
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
    if mel:
        mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result

#Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
#Emotions to observe
observed_emotions=['neutral','calm', 'happy', 'fearful','sad','disgust']

def load_data(test_size=0.25):
    x,y=[],[]
    for file in glob.glob(r"F:\Stuti project\Ravdess small sample datase\\Actor_*\\*.wav"):
        try:
            
            file_name = os.path.basename(file)
            emotion = emotions[file_name.split("-")[2]]
            if emotion not in observed_emotions:
                continue
            X, sample_rate = librosa.load(file)
            feature = extract_feature(X,sample_rate, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
        except ValueError:
            continue
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

x_train,x_test,y_train,y_test = load_data(test_size=0.25)
print(x_train,x_test,y_train,y_test)
