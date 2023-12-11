import pandas as pd
import numpy as np
import wfdb
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, iirnotch
import os

class ECGDataset(Dataset):
    def __init__(self, csv_path, ecg_data_path, split):
        self.patients_data = pd.read_csv(csv_path, index_col='ecg_id')  # dataframe of metadata
        self.ecg_data_path = ecg_data_path
        self.ecg_signals, self.ids = self.load_ecg_signals() # numpy.ndarray of ecg records, and ids
        self.split = split

    def load_ecg_signals(self):
        cache_path = 'cache' 
        # make dir
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        # check if already exist
        signals_file = os.path.join(cache_path, 'ecg_signals.npy')
        ids_file = os.path.join(cache_path, 'ids.npy')
        if os.path.exists(signals_file) and os.path.exists(ids_file):
            ecg_signals = np.load(signals_file)
            ids = np.load(ids_file)
        else:
            ecg_signals = []
            ids = []
            for index, patient_info in self.patients_data.iterrows():
                filename_lr = patient_info['filename_lr']
                record = wfdb.rdrecord(f"{self.ecg_data_path}/{filename_lr}")
                ecg_signal = record.p_signal[:, [0]] # for now we select only one lead
                ecg_signal = self.filter_signal(ecg_signal)
                ecg_signals.append(ecg_signal)
                ids.append(patient_info['patient_id'])

            ecg_signals = np.array(ecg_signals)
            ecg_signals = np.swapaxes(ecg_signals, 1, 2)
            ids = np.array(ids)

            # save on disk
            np.save(signals_file, ecg_signals)
            np.save(ids_file, ids)
        return ecg_signals, ids

    def cut_signal(self, signal):
        # Taglia il segnale a metà
        midpoint = signal.shape[2] // 2
        part1, part2 = signal[:, :, :midpoint], signal[:, :, midpoint:]
        return part1, part2

    def __len__(self):
        return len(self.ecg_signals)

    def __getitem__(self, index):
        ecg_signal = self.ecg_signals[index, :, :]
        patient_id = self.ids[index]

        # Usa la funzione cut_signal per dividere il segnale
        part1, part2 = self.cut_signal(ecg_signal)

        # Restituisci parti del segnale basate sulla modalità di split
        if self.split == 'first_half':
            return torch.tensor(part1).type(torch.float), torch.tensor(patient_id) #dava problemi
            #return torch.from_numpy(part1.copy()).type(torch.float32), patient_id
        elif self.split == 'second_half':
            #return torch.from_numpy(part2.copy()).type(torch.float32), patient_id
            return torch.tensor(part2).type(torch.float), torch.tensor(patient_id)

    def filter_signal(self, signal, fs=100):
        [b, a] = butter(3, (0.5, 40), btype='bandpass', fs=fs)
        signal = filtfilt(b, a, signal, axis=0)
        [bn, an] = iirnotch(50, 3, fs=fs)
        signal = filtfilt(bn, an, signal, axis=0)
        return signal
    
#train_dataset = ECGDataset("primoecg.csv", "archive/", split='first_half')
