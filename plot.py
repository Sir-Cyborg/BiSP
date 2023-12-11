import matplotlib.pyplot as plt
import numpy as np

def show_signals(signal, id, pred, title='ECG Signal'):
    signal = signal.numpy()  # Converte il tensore PyTorch in un array NumPy

    num_leads = signal.shape[1]
    # Crea una figura con 3 assi (grafici)
    fig, axs = plt.subplots(3, 1, figsize=(10, 20), sharex=True, sharey=True)

    for i in range(3):
        for j in range(num_leads):
            axs[i].plot(signal[i, j, :])  # Utilizzo di signal[i, 0, :] per accedere ai dati
            axs[i].set_title(f'Segnale ECG {i}, True:{id[i]}, Pred:{pred[i]}')

    fig.suptitle(title, y=0.92)  # Aggiunge un titolo alla figura
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()
