import pandas as pd
import numpy as np
import wfdb
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, iirnotch

import plot

class ECGDataset(Dataset):
    def __init__(self, csv_path, ecg_data_path, split) -> None:
        self.patients_data = pd.read_csv(csv_path, index_col='ecg_id')  #il dataframe
        self.ecg_data_path = ecg_data_path
        self.ecg_signals, self.ids = self.load_ecg_signals() #super lista #numpy.ndarray
        self.split = split

    def load_ecg_signals(self):
        ecg_signals = []
        ids = []
        for index, patient_info in self.patients_data.iterrows():
            filename_lr = patient_info['filename_lr']
            record = wfdb.rdrecord(f"{self.ecg_data_path}/{filename_lr}")
            ecg_signal = record.p_signal[:, [0]] 
            ecg_signal = self.filter_signal(ecg_signal)
            ecg_signals.append(ecg_signal)
            ids.append(patient_info['patient_id'])

        ecg_signals = np.array(ecg_signals)
        ecg_signals = np.swapaxes(ecg_signals, 1, 2)
        ids = np.array(ids)
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
            return torch.tensor(part1).type(torch.float), patient_id #dava problemi
            #return torch.from_numpy(part1.copy()).type(torch.float32), patient_id
        elif self.split == 'second_half':
            #return torch.from_numpy(part2.copy()).type(torch.float32), patient_id
            return torch.tensor(part2).type(torch.float), patient_id

    def filter_signal(self, signal, fs=100):
        [b, a] = butter(3, (0.5, 40), btype='bandpass', fs=fs)
        signal = filtfilt(b, a, signal, axis=0)
        [bn, an] = iirnotch(50, 3, fs=fs)
        signal = filtfilt(bn, an, signal, axis=0)
        return signal
