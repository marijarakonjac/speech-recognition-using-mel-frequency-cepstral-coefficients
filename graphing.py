import os
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter

def convert_to_wav(file_path):
    wav_path = os.path.splitext(file_path)[0] + ".wav"
    if not os.path.exists(wav_path):
        audio = AudioSegment.from_file(file_path)
        audio.export(wav_path, format="wav")
    return wav_path

def butterworth_filter(y, fs):
    b, a = butter(N=6, Wn=[300.0, 4000.0], fs=fs, btype='band')
    filtered_signal = lfilter(b, a, y)
    return filtered_signal

def plot_signal(y, fs):
    t = np.arange(0, len(y)) / fs
    length = len(y) / fs

    plt.figure()
    plt.plot(t, y)
    plt.title('Vremenski oblik signala')
    plt.ylabel('y(t)')
    plt.xlabel('t[s]')
    plt.grid(True)
    plt.xlim(0, length)
    plt.show()

def plot_spectrogram(y, fs):
    t = np.arange(0, len(y)) / fs
    length = len(y) / fs

    plt.figure()
    plt.specgram(y, Fs=fs, NFFT=1024, noverlap=512, cmap='viridis', scale='dB')
    plt.title('Spektrogram signala')
    plt.ylabel('f[Hz]')
    plt.xlabel('t[s]')
    plt.xlim(0, length)
    plt.grid(True)
    plt.colorbar()
    plt.show()

def plot_amplitude_spectrum(y, fs):
    n = len(y)
    Y = np.fft.fft(y)
    Y = np.abs(Y[:n // 2])
    freqs = np.fft.fftfreq(n, 1 / fs)[:n // 2]

    plt.figure()
    plt.plot(freqs, Y)
    plt.title('Amplitudski spektar signala')
    plt.xlabel('f[Hz]')
    plt.ylabel('Amplituda')
    plt.grid(True)
    plt.show()

file_path = 'C:\\Users\\Marija\\Documents\\diplomski\\baza_Nikolina\\Petra Miljkovic - dete\\signal\\New Recording 212.m4a'
wav_path = convert_to_wav(file_path)
y, fs = librosa.load(wav_path, sr=None)

plot_signal(y, fs)
plot_spectrogram(y, fs)
plot_amplitude_spectrum(y, fs)

wl = int(20e-3 * fs)
E = np.zeros(np.size(y))
Z = np.zeros(np.size(y))

for i in range(int(wl), len(y)):
    rng = np.arange(i - int(wl) + 1, i)
    E[i] = np.sum(y[rng] ** 2)
    Z[i] = np.sum(np.abs(np.sign(y[rng + 1]) - np.sign(y[rng])))

Z = Z / 2 / wl

t = np.arange(0, len(y)) / fs

plt.figure()
plt.plot(t, y, label='signal')
plt.plot(t, E, color='red',label='kve')
plt.grid(True)
plt.title('Kratkovremenska energija')
plt.xlabel('t[s]')
plt.ylabel('E')
plt.legend()
plt.show()

plt.figure()
plt.plot(t, y, label='signal')
plt.plot(t, Z, color='red', label='zcr')
plt.grid(True)
plt.title('Brzina prolaska kroz nulu')
plt.xlabel('t[s]')
plt.ylabel('Z')
plt.legend()
plt.show()
