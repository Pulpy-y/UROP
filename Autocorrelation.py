#!/usr/bin/env python
# coding: utf-8

# In[331]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import librosa
import librosa.display

# 1. Get the file path to the audio example
filename = "A. Hume, P. Livingstone - A Guid New Year.wav"

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load(filename, sr=None)
N = y.size
print(sr)
music_total_length = N/sr


# In[332]:


def norm_corr(x,y):
    N = x.size + y.size - 1
    # normalize x and y respectively 
    normx = np.linalg.norm(x)
    x1 = x/normx
    normy = np.linalg.norm(y)
    y1 = y/normy
    # pad zeros in front to make the length x+y-1
    x1=np.concatenate([x1, np.zeros(N-x.size)])
    y1 = np.concatenate([y1, np.zeros(N-y.size)])
    #x1 = np.pad(x1, (N-1-x.size, 0), 'constant', constant_values=(0, 0))
    #y1 = np.pad(y1, (N-1-y.size, 0), 'constant', constant_values=(0, 0))
    xfft = np.fft.fft(x1)
    yfft = np.fft.fft(y1)
    corr = np.fft.ifft(np.multiply(np.conjugate(xfft), yfft))

    return corr


# In[333]:


#pad y with N-1 zeros in front 
y_pad = np.pad(y, (N-1, 0), 'constant', constant_values=(0, 0))

print("number of points after padding:",y_pad.size) #after padding N zeros in front 


# In[321]:


# Perform autocorrelation on y to find repeating locations
ncorr = norm_corr(y,y)

# Show only the left half (positive x) to compare with spectrogram 
corr_pos = ncorr[0:N]

t = np.arange(0, N/sr, 1/sr)

# Figure size can be changed here
plt.figure(figsize=(8, 6))

# Show all integer time values
plt.xticks(np.arange(min(t), max(t)+1, 5.0))
plt.plot(t, corr_pos)


# In[322]:


# Find peaks in autocorrelation result
# using signal package 
distance = 44100
peaks, properties = signal.find_peaks(corr_pos, distance=distance, height=0)
heights = properties['peak_heights']
second_height = heights[0]
h_thres = second_height * 0.3

peaks = signal.find_peaks(corr_pos, distance=distance, height=h_thres)[0]/sr
peaks_song = peaks
print(peaks_song)


# plot peaks 
plt.figure(figsize=(16, 8))
plt.title('Scipy peak picking (distance=%2.0f samples, Fs=%3.0f, height >%0.3f)'%(distance, sr,h_thres))
plt.xticks(np.arange(min(t), max(t)+1, 5.0))
plt.plot(t, corr_pos)
plt.vlines(peaks_song, 0, 1, color='r', linestyle=':', linewidth=1);


# In[304]:


print(heights)


# In[209]:


def repeatsAt(y_sample, y_whole, sr, threshold):
    corr = norm_corr(y_sample, y)
    t = np.arange(0, corr.size/sr, 1/sr)

    # Get the repeating positions
    distance = sr
    # discard the extra length (== sample segment length) of corr result 
    corr_song = corr[:corr.size-y_sample.size]
    peaks, properties = signal.find_peaks(corr_song, distance=distance, height = threshold)
    highest_peak = properties["peak_heights"].max()
    return [peaks, highest_peak]


# In[305]:


print(peaks_song)


# In[32]:


print(repeatsAt(y_peak, y, sr, thres)[0])


# In[330]:


# if we set the window to be 8 seconds 
# use the duration from 8 to 16 seconds
t_start = 0
t_diff = 8
#y_pad = np.concatenate([y, np.zeros(t_diff*sr)])
y_peak = y[int(t_start*sr):int((t_start+t_diff)*sr)]
print(y_peak.size)

corr = norm_corr(y_peak, y) 
t = np.arange(0, (corr.size - y_peak.size)/sr, 1/sr)
music_total_length = (corr.size - y_peak.size)/sr
corr2 = corr[:corr.size - y_peak.size]
plt.figure(figsize=(16, 8))
plt.xticks(np.arange(min(t), max(t)+1, 16.0))
plt.plot(t, corr2)

# Get the repeating seconds
distance = 44100
peak_loc, properties = signal.find_peaks(corr2, distance=distance, height=0)
peaks = peak_loc/sr
heights = properties["peak_heights"]
thres = heights.max() * 0.65

peak_loc, properties = signal.find_peaks(corr2, distance=distance, height=thres)
peaks = peak_loc/sr
print(properties["peak_heights"])


title='Scipy peak picking (distance=%2.0f samples, Fs=%3.0f)'%(distance, sr)
print(peaks)
plt.figure(figsize=(16, 8))
plt.xticks(np.arange(min(t), max(t)+1, 16.0))
plt.plot(t, corr2)
plt.vlines(peaks, -1, 1, color='r', linestyle=':', linewidth=1);


# In[326]:


peaks_song = peaks_song[2:]


# In[327]:


print(peaks_song)


# In[37]:


print(thres)


# In[328]:


sr = 44100
prev_p = -1
prev_dur = 0
segments = []
repeat_at = []
for p in peaks_song:
    # discard if overlaps with previous segment detected
    if p < prev_p + prev_dur:
        continue
    
    print("segment starting at time = ", p, "second")
   
    # starting from 8 seconds duration, increment by 4 subsequently
    duration = 8
    longest = False
    rep = []
    
    # duration should not exceed the distance between adjacent repetitions
    # initialize this to total length of the music 
    min_distance = music_total_length
    
    while not longest and min_distance >= dur :
        print("duration:", duration)
        sample = y[int(p*sr):int((p+duration)*sr)]
        
        # find out what threshold to set 
        rep_loc, highest = repeatsAt(sample, y, sr, threshold=0)
        thres = 0.65 * highest
        
        # detect peaks -> repetition location 
        rep_at = repeatsAt(sample, y, sr, thres)[0]/sr
        
        if rep_at.size == 1:
            # if there is only one peak, 
            # must be at its own position => longest repetition is reached
            longest = True 
            break
        else: 
            if rep_at[0] < p:
            # if any repetition pattern occurs before the peak, disgard them
                rep_at = rep_at[rep_at >= p]
            if rep_at.size == 1:
                break
            print("repeat at:", rep_at)
            duration += 4
            rep = rep_at
            min_distance = rep_at[1] - rep_at[0]
    
    duration -= 4

        
    #record the start and end of repeating segment
    segments.append([p, duration])
    repeat_at.append(rep[1:])
    prev_p = p
    prev_dur = duration 
        
for i in range(len(segments)):
    if len(repeat_at[i]) >0:
        print("segement starting from t = ", segments[i][0],
              "repeats at", repeat_at[i],
             "for", segments[i][1], "seconds")


# In[245]:


print(prev_p)
print(duration)


# In[248]:


segments = segments[:1]
repeat_at = repeat_at[:1]
print(segments)


# In[215]:


print(music_total_length)


# In[329]:


# Plot the repetition using bar 
from random import randint

n_seg = len(segments)
fig, ax = plt.subplots()

plt.xlim(right = music_total_length)
ax.set_yticklabels([])

for i in range(len(segments)):
    c = "C"+str(i) # color of this repetition pattern
    s = segments[i]
    dur = s[1]
    if len(repeat_at[i]) >= 1 :
        ax.barh(0.4*i, dur, height = 0.3, color=c, left=s[0],edgecolor="k", align='edge')
   
        for loc in repeat_at[i]:
            ax.barh(0.4*i, dur, height = 0.3, color=c,left=loc,edgecolor="k", align='edge')
    
plt.gcf().set_size_inches(10, 2)
plt.show()


# In[ ]:




