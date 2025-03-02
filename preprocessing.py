import os
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
import librosa
import sounddevice as sd
from scipy.signal import butter
import soundfile as sf

#Dohvata fajlove sa zadatom ekstenzijom u okviru target direktorijuma
def find_files(root_dir, target_dirs, extensions):
    files_list = []
    for root, dirs, files in os.walk(root_dir):
        if os.path.basename(root) in target_dirs:
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    files_list.append(file_path)
    return files_list

#Konvertuje audio fajl u wav
def convert_to_wav(file_path):
    wav_path = os.path.splitext(file_path)[0] + ".wav"
    if not os.path.exists(wav_path):
        audio = AudioSegment.from_file(file_path)
        audio.export(wav_path, format="wav")
    return wav_path

#Dodaje gausovski sum audio signalu
def add_gaussian_noise(audio, variance=0.002):
    samples = np.array(audio.get_array_of_samples())
    noise = np.random.normal(0, variance, samples.shape)
    noisy_samples = samples + noise
    noisy_samples = np.clip(noisy_samples, -32768, 32767)
    return noisy_samples.astype(np.int16)

#Konvertuje audio fajl u wav plus dodaje gaysovski sum
def convert_to_wav1(file_path, target_sample_rate=12000, noise_variance=0.005):
    wav_path = os.path.splitext(file_path)[0] + ".wav"
    if not os.path.exists(wav_path):
        audio = AudioSegment.from_file(file_path)
        noisy_samples = add_gaussian_noise(audio, noise_variance)
        noisy_audio = audio._spawn(noisy_samples)
        noisy_audio.export(wav_path, format="wav")
    return wav_path

'''
root_directory = 'C:\\Nina\\ETF\\DIPLOMSKI\\Baza podataka - Copy\\baza_Nikolina\\Nikolina Ilic'
target_directories = ['grozdje']
extensions = ['.m4a','.mp3']

files = find_files(root_directory, extensions=extensions, target_dirs=target_directories)
print(f"Ukupan broj učitanih fajlova: {len(files)}")

wav_files = [convert_to_wav1(file) for file in files]
print(f"Ukupan broj konvertovanih fajlova: {len(wav_files)}")
'''

#Filtriranje signala
def butterworth_filter(y, fs):
    b, a = butter(N=6, Wn=[300.0, 4000.0], fs=fs, btype='band')
    filtered_signal = lfilter(b, a, y)
    return filtered_signal

#Iscrtavanje vremenskog oblika signala
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

#Iscrtavanje spektrograma
def plot_spectrogram(y, fs):
    t = np.arange(0, len(y)) / fs
    length = len(y) / fs

    plt.figure()
    plt.specgram(y, Fs=fs)
    plt.title('Spektrogram signala')
    plt.ylabel('f[Hz]')
    plt.xlabel('t[s]')
    plt.xlim(0, length)
    plt.grid(True)
    plt.colorbar()
    plt.show()


#Iscrtavanje amplitudskog spektra
def plot_amplitude_spectrum(y, fs):
    n = len(y)
    Y = np.fft.fft(y)
    Y = np.abs(Y[:n // 2])
    freqs = np.fft.fftfreq(n, 1 / fs)[:n // 2]

    plt.figure()
    plt.plot(freqs, Y)
    plt.title('Amplitudski spektar signala')
    plt.xlabel('Frekvencija (Hz)')
    plt.ylabel('Amplituda')
    plt.grid(True)
    plt.show()

#Segmentracija reci u audio signalu
def word_segmentation(y, fs, wl):
    E = np.zeros(np.size(y))
    Z = np.zeros(np.size(y))

    for i in range(int(wl), len(y)):
        rng = np.arange(i - int(wl) + 1, i)
        E[i] = np.sum(y[rng] ** 2)
        Z[i] = np.sum(np.abs(np.sign(y[rng + 1]) - np.sign(y[rng])))

    Z = Z / 2 / wl

    ITU = 0.05 * np.max(E)
    ITL = 0.00001 * np.max(E)
    words_start = []
    words_end = []

    for i in range(1, len(E)):
        if (E[i - 1] < ITU) and (E[i] >= ITU):
            words_start.append(i)
        if (E[i - 1] > ITU) and (E[i] <= ITU):
            words_end.append(i)

    if not words_start or not words_end:
        return y

    word_start = words_start[0]
    word_end = words_end[0]

    shift1 = word_start - 1
    while shift1 > 0 and E[shift1] > ITL:
        shift1 -= 1
        word_start = shift1 + 1

    shift1 = word_end - 1
    while shift1 < len(E) - 1 and E[shift1] > ITL:
        shift1 += 1
        word_end = shift1 + 1

    word = np.zeros((len(y), 1))
    word[word_start - 1:word_end] = max(y) * np.ones((word_end - word_start + 1, 1))

    t = np.arange(0, len(y)) / fs
    plt.figure()
    plt.plot(t, y, 'b')
    plt.plot(t, word/2, 'r')
    plt.title('Prikaz segmentirane reci')
    plt.legend(['vremenski oblik', 'rec'])
    plt.xlabel('t[s]')
    plt.grid(True)
    plt.show()

    t = np.arange(0, len(y[word_start-1:word_end])) / fs
    plt.figure()
    plt.plot(t, y[word_start-1:word_end])
    plt.title("Izdvojena rec")
    plt.xlabel('t[s]')
    plt.grid(True)
    plt.show()

    t = np.arange(1 / fs, (len(E) + 1) / fs, 1 / fs)
    plt.figure()
    plt.plot(E)
    plt.plot(y)
    plt.grid(True)
    plt.title('Kratkovremenska energija originalnog signala')
    plt.xlabel('t[s]')

    plt.figure()
    plt.plot(Z)
    plt.plot(y)
    plt.grid(True)
    plt.title('Brzina prolaska kroz nulu')
    plt.xlabel('t[s]')
    plt.show()

    return y[word_start-1:word_end]

#Pustanje audio signala
def play_signal(y, fs):
    print("Pocetak...")
    sd.play(y, fs)
    sd.wait()
    print("Kraj")


# Segmentacija i cuvanje novih .wav fajlova.
'''
output_folder = 'C:\\Users\\Marija\\PycharmProjects\\diplomski\\kuca'
counter = 495
for audio_file in files:
    y, fs = librosa.load(audio_file, sr=None)
    wl1 = int(20e-3 * fs)

    # plot_signal(y,fs)
    # plot_spectrogram(y, fs)
    # plot_amplitude_spectrum(y,fs)

    # Segmentacija i cuvanje novih fajlova.
    y = butterworth_filter(y,fs)
    y = word_segmentation(y, fs, wl1)
    # play_signal(y, fs)

    # Generisanje novog imena fajla i čuvanje u output folder
    output_file_path = os.path.join(output_folder, f'kuca_{counter}.wav')
    sf.write(output_file_path, y, fs)
    print(f'Saved processed audio to {output_file_path}')

    # Povećajte brojač za sledeći fajl
    counter += 1
'''

