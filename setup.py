import streamlit as st
import numpy as np
import librosa, librosa.display
import tensorflow as tf
import os
import cv2
from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt
from keras.models import load_model
from skimage import transform


st.set_page_config(
     page_title="SER Web App",
     layout="wide",
     initial_sidebar_state="collapsed"
 )


# improved_model
# improved_model = load_model("improved.hdf5")
def createWaveplot(sample, sr, fig_size, algo):
  plt.figure(figsize=fig_size)
  librosa.display.waveplot(sample, sr)
  plt.title("Audio Waveplot")
  plt.xlabel("Time (seconds)")
  plt.ylabel("Amplitude")

  if(algo == 0): #0 is improved 1 = baseline
        plt.savefig('audio_waveplot_0.png',transparent=True,bbox_inches='tight', dpi=72)
        # st.title("Audio Waveplot")
        st.markdown("<h4 style='text-align: center;'>Audio Waveplot</h4>", unsafe_allow_html=True)
        st.image('audio_waveplot_0.png', caption=' ')
  else:
        plt.savefig('audio_waveplot_1.png',transparent=True,bbox_inches='tight', dpi=72)
        # st.title("Audio Waveplot")
        st.markdown("<h4 style='text-align: center;'>Audio Waveplot</h4>", unsafe_allow_html=True)
        st.image('audio_waveplot_1.png', caption=' ')

def create_melspectrogram(sample, srate, fig_size, algo):
    plt.figure(figsize=fig_size)
    mel_spectrogram = librosa.feature.melspectrogram(sample, sr=srate, n_fft=2048, hop_length=128, n_mels=256)
    mel_spect = librosa.power_to_db(mel_spectrogram, ref=np.max) 
    mel_spect_resize = cv2.resize(mel_spect, (256, 256)) 
    librosa.display.specshow(mel_spect_resize,fmax=8000)

    if(algo == 0): #0 is improved 1 = baseline
        plt.savefig('mel_spectrogram_0.png',transparent=True,bbox_inches='tight',pad_inches=0, dpi=72)
        # st.title("Mel-Spectrogram")
        st.markdown("<h4 style='text-align: center;'>Mel-Spectrogram</h4>", unsafe_allow_html=True)
        st.image('mel_spectrogram_0.png', caption=' ')
    else:
        plt.savefig('mel_spectrogram_1.png',transparent=True,bbox_inches='tight',pad_inches=0, dpi=72)
        # st.title("Mel-Spectrogram")
        st.markdown("<h4 style='text-align: center;'>Mel-Spectrogram</h4>", unsafe_allow_html=True)
        st.image('mel_spectrogram_1.png', caption=' ')
   

def data_visual(audiofile, algo):
    fig_size = (10,5)
    sample, srate = librosa.load(audiofile)
    createWaveplot(sample, srate, fig_size, algo)
    create_melspectrogram(sample, srate, fig_size, algo)

def label(index):
  switch={
    '0':'Angry',
    '1':'Disgust',
    '2':'Fear',
    '3':'Happy',
    '4':'Neutral',
    '5':'Sad',
    '6':'Surprise'
  }
  return switch.get(index)

def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

#Modified Algo
def classify_modified(img_path):
    model = load_model("DSCNN_ELM500epochsearlystop.h5")
    vector_prediction = model.predict(load_image(img_path))

    return vector_prediction

def probabilities_modified(img_path):
    probabilities = tf.nn.softmax(classify_modified(img_path)).numpy()

    return probabilities

def modified_predicted_emotion(img_path):
    result =  np.argmax(probabilities_modified(img_path))
    emotion = label(f'{result}')

    return emotion
  
def classify(img_path):
    #baseline model
    model = load_model("SER_500_epochs_model.h5")
    vector_prediction = model.predict(load_image(img_path))

    return vector_prediction

def probabilities(img_path):
    probabilities = tf.nn.softmax(classify(img_path)).numpy()

    return probabilities

def baseline_predicted_emotion(img_path):
    result =  np.argmax(probabilities(img_path))
    emotion = label(f'{result}')

    return emotion

def get_actual_emotion(file_name):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    for i in emotions:
        if i in file_name: 
            file = i
    
    return file
    

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

#     
#     st.write(prediction)
#     st.write(score)
#     print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )


