
# coding: utf-8

# In[1]:


import os
from pathlib import Path
import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


# In[13]:


for j in range(10) :
    for i in range(50) :
        filepath = './recordings/' + str(j) + '_theo_' + str(i) + '.wav'
        sample_rate, samples = wavfile.read(filepath)
        freqs, times, spectrogram = log_specgram(samples, sample_rate)
        mean = np.mean(spectrogram, axis = 0)
        std = np.std(spectrogram, axis = 0)
        spectrogram = (spectrogram - mean) / std
        fig = plt.figure(figsize=(14, 10))

        ax2 = fig.add_subplot(212)
        ax2.imshow(spectrogram.T, aspect='auto', origin='lower', 
               extent=[times.min(), times.max(), freqs.min(), freqs.max()])
        ax2.set_yticks(freqs[::16])
        ax2.set_xticks(times[::16])
        ax2.set_title('Spectrogram of 0_jackson_1.wav')
        ax2.set_ylabel('Freqs in Hz')
        ax2.set_xlabel('Seconds')

        savepath = './data/' + str(j) + '_theo_' + str(i) + '.png'
        fig.savefig(savepath, dpi = fig.dpi)

        croppath = './crop/' + str(j) + '_theo_' + str(i) + '.png'
        img = Image.open(savepath)
        cropped = img.crop((127, 384, 906, 629))
        cropped.save(croppath)


# In[ ]:




