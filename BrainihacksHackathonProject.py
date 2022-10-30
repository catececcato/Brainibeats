from pylsl import StreamInlet, resolve_stream
import numpy as np
import pandas as pd
pitchlili = []
import math
from pythonosc import osc_message_builder
from pythonosc import udp_client
from numpy import mean
from scipy.fft import fft
import mne_connectivity as mc
import scipy
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne_connectivity import spectral_connectivity_epochs
from mne.datasets import sample
from scipy.signal import butter, lfilter
from scipy.signal import butter, sosfilt, sosfreqz
import warnings

warnings.filterwarnings("ignore")

#Initialising variables
k = 0
h = 0
j = -1
pitchlist = []

# initialize the streaming layer
finished = False
streams = resolve_stream()
inlet = StreamInlet(streams[0])


#Setting sample frequency
n_samples = 250


# initialize the colomns of data and dictionary to capture the data.
columns=['Time','FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8','AccX','AccY','AccZ','Gyro1','Gyro2','Gyro3', 'Battery','Counter','Validation']
data_dict = dict((k, []) for k in columns)

#Keeps running until finished is True
while not finished:
   # Get the streamed data.
   # concatenate timestamp and data in 1 list
   data, timestamp = inlet.pull_sample()
   all_data = [timestamp] + data
   
#Create function for bandpass filtering
   def butter_bandpass(signal):
        #Sampling frequency
        fs = 250
        
        #Nyquist frequency
        nyq = 0.5 * fs
 
        #Lower bandpass threshold
        lowcut = 1.0
        low = lowcut / nyq
        #Upper bandpass threshold
        highcut =35.0
        high = highcut / nyq
        
        #Order of filtering (2 is chosen for processing speed)
        order = 2
        
        #Create Butterworth filter
        b, a = scipy.signal.butter(order, [low, high], "bandpass", analog = False)
        #Apply butterworth filter to data
        y = scipy.signal.filtfilt(b, a, signal, axis = 0)
        #Returns bandpassed filtered data
        return y
        

#Create function for Frontal Alpha Asymmetry (Measure for valence)   
   def FrontalAlphaAsymmetry(left, right):
    lefthem = np.array(left)
    lefthem = lefthem[~np.isnan(lefthem)]
    righthem = np.array(right)
    righthem = righthem[~np.isnan(righthem)]
    #print("left hem: ", lefthem)
    #print("right hem: ", righthem)
    FFTLefthem = np.fft.fft(lefthem)
    
    FFTRighthem = np.fft.fft(righthem)
    frequenciesR = np.fft.fftfreq(n_samples) * n_samples * 1 / (1-0)
    frequenciesL = np.fft.fftfreq(n_samples) * n_samples * 1 / (1-0)
    FFTLefthem = np.mean(FFTLefthem[8:12].real)
    FFTRighthem = np.mean(FFTRighthem[8:12].real)
    FFTLefthem = np.log(FFTLefthem)
    FFTRighthem = np.log(FFTRighthem)
    
    FrontalAlpha = FFTRighthem - FFTLefthem
    return FrontalAlpha
   
   # updating data dictionary with newly transmitted samples
  

   i = 0
   for keyz in list(data_dict.keys()):
      data_dict[keyz].append(all_data[i])
      i = i + 1
   j += 1
   if j >= 250:
     
    df = pd.DataFrame.from_dict(data_dict)
    
        
    #selecting the columns relevant to the channels
    cols = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    

    #create a new column with the average value for each channel
    df["average"] = df[cols].mean(axis=1)
    #df["average"] = df.mean(axis=1)
    
    #create data frame with 250 samples
    dada = df["average"][j-250:j]
    #turn the dataframe into an array to make it the input for the fourier transform
    numnum = np.array(dada)
    #fourier transform
    FFTSamples = np.fft.fft(numnum)
    #determine amplitude of waves obtained after fourier transform
    amplitudes = 2 / n_samples * np.abs(FFTSamples.real) 
    #print(amplitudes)
    #FAA to determine the exponent of the power-law exponent index         
    FAA = FrontalAlphaAsymmetry(df["C3"][j-250: j],df["C4"][j-250:j])
    #pitch calculated according to Fechner's law
   # print(FAA)
    if math.isnan(FAA):
        FAA = 1
    else:
        FAA = FAA*0.8
    Pitch = (((-40/FAA)*np.log(np.mean(amplitudes[1:35]))/100)+10)

    #collect list of pitches for normalization
    if j <= 10000:
        pitchlili.append(Pitch)
        
    mini = min(pitchlili)
    maxi = max(pitchlili)
    #normalization
    Pitch = (Pitch - mini)/ (maxi-mini)
    if not math.isnan(Pitch):
        Pitch = round(Pitch* 100 + 20)
        pitchlist.append(Pitch)
    #print(FFTSamples)
    #print(np.mean(amplitudes[8:12]))
    #print(np.log(np.mean(amplitudes[8:12])))
    #print("here is the pitch: ", Pitch)
    #print(pitchlist)
    import numpy as np
    import scipy.signal as sig
    
  
  
   
    
    #bandpass applied to limit the signal to EEG data    
    yee = butter_bandpass(np.array(df["FZ"]))
    yee2 = butter_bandpass(np.array(df["C3"]))
    yee3 = butter_bandpass(np.array(df["CZ"]))
    yee4 = butter_bandpass(np.array(df["C4"]))
    yee5 = butter_bandpass(np.array(df["PZ"]))
    yee6 = butter_bandpass(np.array(df["PO7"]))
    yee7 = butter_bandpass(np.array(df["OZ"]))
    yee8 = butter_bandpass(np.array(df["PO8"]))#'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8'
    #update data based on filtered data
    df["FZ"] = yee
    df["C3"] = yee2
    df["CZ"] = yee3
    df["C4"] = yee4
    df["PZ"] = yee5        
    df["PO7"] = yee6
    df["OZ"] = yee7
    df["PO8"] = yee8
    
        #butter_bandpass_filter(df["FZ"][j-500:j])
    #select 250 samples in two channels to detect sybchrony between brain regions
    y1 = df["FZ"][j-250:j]
    y2 = df["PO8"][j-250:j]
    #indices = ["y1", "y2"]
    #print(y2)
    #function to get the Phase Locking Value (PLV) 
    def hilphase(y1,y2):
        sig1_hill=sig.hilbert(y1)
        sig2_hill=sig.hilbert(y2)
        pdt=(np.inner(sig1_hill,np.conj(sig2_hill))/(np.sqrt(np.inner(sig1_hill,
                np.conj(sig1_hill))*np.inner(sig2_hill,np.conj(sig2_hill)))))
        phase = np.abs(np.angle(pdt))
        #print(phase)

        return phase
    
    k += 1
    #every 250 samples, check PLV and produce sounds based on the notes
    if k == 250:
        #print(df)
        #print(round(mean(pitchlist)))
        note = round(mean(pitchlist))
        #soundlist = [62,64,66,67,69,71,73,74,62,64,66,67,69,71,73,74]
        phase = hilphase(y1, y2)

        sender = udp_client.SimpleUDPClient("192.168.201.106", 4560)
        #sender.send_message('/trigger/prophet', [61+note, 100, 1,61+note+6  ])
        a = note
        #adjustement of notes based on synchrony 
        if round(phase,2) < 0.05:
            sender.send_message('/trigger/prophet', [a,a+12 ])
            print("base: ", a, " second: ", a+12)
        elif round(phase,2) < 0.1:
           sender.send_message('/trigger/prophet', [a,a+5 ])
           print("base: ", a, " second: ", a+5)
        elif round(phase,2) < 0.15:
           sender.send_message('/trigger/prophet', [a,a+8 ])
           print("base: ", a, " second: ", a+8)
        else:
           sender.send_message('/trigger/prophet', [a, a+11 ])
           print("base: ", a, " second: ", a+11)
        
        #sender.send_message('/trigger/prophet', [soundlist[note+2], 100, 1])
        #sender.send_message('/trigger/prophet', [soundlist[note+4], 100, 1])
        pitchlist = []
        k = 0
    #df["C3"]
   #print(data_dict)
   # data is collected at 250 Hz. Let's stop data collection after 10 seconds. Meaning we stop when we collected 250*10 samples.
   if (len(data_dict['Time']) >= 250*60):
      finished = True
      
      

# lastly, we can save our data to a CSV format.
data_df = pd.DataFrame.from_dict(data_dict)
#data_df.to_csv('EEGdata.csv', index = False)