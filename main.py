#%%
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

#Funkcija za dohvatanje fajlova
def find_files(root_directory, extensions, target_dirs):
    files = []
    for target_dir in target_dirs:
        full_path = os.path.join(root_directory, target_dir)
        for root, _, filenames in os.walk(full_path):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in extensions):
                    files.append(os.path.join(root, filename))
    return files

#Filtriranja preemfazis filtrom
def pre_emphasis_filter(y, coefficient=0.97):
    emphasized_signal = np.append(y[0], y[1:] - coefficient * y[:-1])
    return emphasized_signal

#Normalizacija signala
def normalize_signal(y, target_variance=1.0):
    current_variance = np.var(y)
    normalized_signal = y * np.sqrt(target_variance / current_variance)
    return normalized_signal

#Izracunavanje kepstralnih koeficijenata
def calculate_mfcc(y, sr, n_mfcc=15):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=len(y), hop_length=len(y)+1)
    return mfccs.flatten() #da bi se vratila samo lista koeficijenata,a ne lista listi

#Procesiranje signala
def process_audio_files(files, num_parts=1):
    features = []
    for file in files:
        y, fs = librosa.load(file, sr=None)
        y = librosa.resample(y, orig_sr=fs, target_sr=8000)
        y = pre_emphasis_filter(y)
        y = normalize_signal(y, 1.0)

        part_length = len(y) // num_parts
        curr_features = []
        for i in range(num_parts):
            start = i * part_length
            end = (i + 1) * part_length if i < num_parts - 1 else len(y)
            y_part = y[start:end]

            mfccs = calculate_mfcc(y_part, sr=8000)
            curr_features.append(mfccs)
        features.append(list(np.array(curr_features).flatten()))
    return features

#%%
#Primer na jednom signalu
file_path = 'C:\\Users\\Marija\\PycharmProjects\\diplomski\\Baza bez dece\\grozdje_r\\grozdje_0.wav'
y, fs = librosa.load(file_path, sr=None)

y = librosa.resample(y, orig_sr=fs, target_sr=8000)
y = pre_emphasis_filter(y)
y = normalize_signal(y)
mfccs = calculate_mfcc(y, sr=8000)

#Vizuelizacija kepstralnih koeficijenata
plt.figure()
if mfccs.ndim == 2 and mfccs.shape[1] > 1:
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
else:
    plt.plot(mfccs.T)
    plt.xlabel('indeks')
    plt.ylabel('vrednost')
plt.title('Kepstralni koeficijenti za reč grožđe')
plt.tight_layout()
plt.show()

#%%
#Sada za sve audio signale
root_directory = 'C:\\Users\\Marija\\PycharmProjects\\diplomski\\Baza bez dece'
extensions = ['.wav']

target_directories_jedan = ['jedan_r']
target_directories_sedam = ['sedam_r']
target_directories_signal = ['signal_r']
target_directories_grozdje = ['grozdje_r']
target_directories_kuca = ['kuca_r']

files_jedan = find_files(root_directory, extensions, target_directories_jedan)
files_sedam = find_files(root_directory, extensions, target_directories_sedam)
files_signal = find_files(root_directory, extensions, target_directories_signal)
files_grozdje = find_files(root_directory, extensions, target_directories_grozdje)
files_kuca = find_files(root_directory, extensions, target_directories_kuca)

signal_features = process_audio_files(files_signal, num_parts = 3)
jedan_features = process_audio_files(files_jedan, num_parts = 3)
sedam_features = process_audio_files(files_sedam, num_parts = 3)
grozdje_features = process_audio_files(files_grozdje, num_parts = 3)
kuca_features = process_audio_files(files_kuca, num_parts = 3)

#%% Test baza
root_directory = 'C:\\Users\\Marija\\PycharmProjects\\diplomski\\Baza test skup sa decom'
extensions = ['.wav']

target_directories_jedan = ['jedan']
target_directories_sedam = ['sedam']
target_directories_signal = ['signal']
target_directories_grozdje = ['grozdje']
target_directories_kuca = ['kuca']

files_jedan_test = find_files(root_directory, extensions, target_directories_jedan)
files_sedam_test = find_files(root_directory, extensions, target_directories_sedam)
files_signal_test = find_files(root_directory, extensions, target_directories_signal)
files_grozdje_test = find_files(root_directory, extensions, target_directories_grozdje)
files_kuca_test = find_files(root_directory, extensions, target_directories_kuca)

signal_features_test = process_audio_files(files_signal_test, num_parts = 3)
jedan_features_test = process_audio_files(files_jedan_test, num_parts = 3)
sedam_features_test = process_audio_files(files_sedam_test, num_parts = 3)
grozdje_features_test = process_audio_files(files_grozdje_test, num_parts = 3)
kuca_features_test = process_audio_files(files_kuca_test, num_parts = 3)

#%%
# Ispisivanje koeficijenata i dimenzija
#def print_feature_sizes(feature_list, label):
    #print(f"Number of {label} features: {len(feature_list)}")
    #for i, feature in enumerate(feature_list[:15]):  # Print first 5 to avoid too much output
        #print(f"Shape of {label} feature {i+1}: {feature.shape}")
        #print(feature[:])

#print_feature_sizes(signal_features, "signal")

#%% KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#Svi feature-i i labele za trening bazu
all_features_train = np.vstack([
    signal_features,
    jedan_features,
    sedam_features,
    grozdje_features,
    kuca_features
])

labels_train = np.array(['signal'] * len(signal_features) +
                        ['jedan'] * len(jedan_features) +
                        ['sedam'] * len(sedam_features) +
                        ['grozdje'] * len(grozdje_features) +
                        ['kuca'] * len(kuca_features))

#Svi feature-i i labele za test bazu
all_features_test = np.vstack([
    signal_features_test,
    jedan_features_test,
    sedam_features_test,
    grozdje_features_test,
    kuca_features_test
])

labels_test = np.array(['signal'] * len(signal_features_test) +
                       ['jedan'] * len(jedan_features_test) +
                       ['sedam'] * len(sedam_features_test) +
                       ['grozdje'] * len(grozdje_features_test) +
                       ['kuca'] * len(kuca_features_test))

#Enkodiranje labela u numericke vrednosti (KNN baca warning ako se ovo ne uradi)
label_encoder = LabelEncoder()
encoded_labels_train = label_encoder.fit_transform(labels_train)
encoded_labels_test = label_encoder.transform(labels_test)

#Radimo GRID SEARCH da proverimo koji hiperparametri su optimalni
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()

#Primena Grid Search-a
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(all_features_train, encoded_labels_train)

#Prikazivanje najboljih parametara i najbolje tacnosti
print("Najbolji parametri: ", grid_search.best_params_)
print("Najbolja kros-validaciona tacnost: {:.2f}".format(grid_search.best_score_))

# Evaluacija na test skupu
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(all_features_test)

#Najbolja tacnost
accuracy = accuracy_score(encoded_labels_test, y_pred)
print(f"Tacnost: {accuracy:.2f}")

#print(classification_report(encoded_labels_test, y_pred, target_names=label_encoder.classes_))

#Konfuziona matrica
cm = confusion_matrix(encoded_labels_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Konfuziona matrica')
plt.colorbar()
classes = label_encoder.classes_
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Prediktovana labela')
plt.ylabel('Tačna labela')
plt.tight_layout()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2. else "black")

plt.show()