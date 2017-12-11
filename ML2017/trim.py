
# coding: utf-8

# In[1]:


import scipy.io.wavfile
import numpy as np

def trim_silence(audio, noise_threshold=150):
    """ Removes the silence at the beginning and end of the passed audio data
    :param audio: numpy array of audio
    :param noise_threshold: the maximum amount of noise that is considered silence
    :return: a trimmed numpy array
    """
    start = None
    end = None

    for idx, point in enumerate(audio):
        if np.any(abs(point) > noise_threshold):
            start = idx
            break

    # Reverse the array for trimming the end
    for idx, point in enumerate(audio[::-1]):
        if np.any(abs(point) > noise_threshold):
            end = len(audio) - idx
            break

    return audio[start:end]


def trim_silence_file(file_path, noise_threshold=150):
    """Accepts a file path, trims the audio and overwrites the original file with the trimmed version.
    :param file_path: file to trim
    :param noise_threshold: the maximum amount of noise that is considered silence
    :return: None
    """
    rate, audio = scipy.io.wavfile.read(file_path)
    trimmed_audio = trim_silence(audio, noise_threshold=noise_threshold)
    scipy.io.wavfile.write(file_path, rate, trimmed_audio)


# In[4]:


for i in range(10):
    filepath = './myvoice/' + str(i) + '.wav'
    trim_silence_file(filepath, )

