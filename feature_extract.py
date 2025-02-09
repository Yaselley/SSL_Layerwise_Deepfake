import librosa
import random
import math
import numpy as np
from scipy import signal
#import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav, series_LA, series_DF

def process_Rawboost_feature(feature, sr=16000, algo=3,nBands = 5,minF = 20,maxF = 8000,minBW = 100,maxBW = 1000,minCoeff = 10,maxCoeff = 100,minG = 0,maxG = 0,minBiasLinNonLin = 5,maxBiasLinNonLin = 20,N_f = 5,P = 10,g_sd = 2,SNRmin = 10,SNRmax = 40):
    # Data process by Convolutive noise (1st algo)
    if algo == 1:
        feature = LnL_convolutive_noise(feature, N_f, nBands, minF, maxF, minBW, maxBW,
                                        minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin,
                                        maxBiasLinNonLin, sr)

    # Data process by Impulsive noise (2nd algo)
    elif algo == 2:

        feature = ISD_additive_noise(feature, P, g_sd)


    # Data process by coloured additive noise (3rd algo)
    elif algo == 3:

        feature = SSI_additive_noise(feature, SNRmin, SNRmax, nBands, minF, maxF, minBW,
                                     maxBW, minCoeff, maxCoeff, minG, maxG, sr)

    # Data process by all 3 algo. together in series (1+2+3)
    elif algo == 4:

        feature = LnL_convolutive_noise(feature, N_f, nBands, minF, maxF, minBW, maxBW,
                                        minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin,
                                        maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, P, g_sd)
        feature = SSI_additive_noise(feature, SNRmin, SNRmax, nBands, minF, maxF, minBW,
                                     maxBW, minCoeff, maxCoeff, minG, maxG, sr)

        # Data process by 1st two algo. together in series (1+2)
    elif algo == 5:

        feature = LnL_convolutive_noise(feature, N_f, nBands, minF, maxF, minBW, maxBW,
                                        minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin,
                                        maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, P, g_sd)

        # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo == 6:

        feature = LnL_convolutive_noise(feature, N_f, nBands, minF, maxF, minBW, maxBW,
                                        minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin,
                                        maxBiasLinNonLin, sr)
        feature = SSI_additive_noise(feature, SNRmin, SNRmax, nBands, minF, maxF, minBW,
                                     maxBW, minCoeff, maxCoeff, minG, maxG, sr)

        # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo == 7:

        feature = ISD_additive_noise(feature, P, g_sd)
        feature = SSI_additive_noise(feature, SNRmin, SNRmax, nBands, minF, maxF, minBW,
                                     maxBW, minCoeff, maxCoeff, minG, maxG, sr)

        # Data process by 1st two algo. together in Parallel (1||2)
    elif algo == 8:

        feature1 = LnL_convolutive_noise(feature, N_f, nBands, minF, maxF, minBW, maxBW,
                                         minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin,
                                         maxBiasLinNonLin, sr)
        feature2 = ISD_additive_noise(feature, P, g_sd)

        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)  # normalized resultant waveform

    # original data without Rawboost processing
    else:

        feature = feature

    return feature

def extract(wav_path, feature_type, aug=None):
    if feature_type == "fft":
        return extract_fft(wav_path, aug)
    if feature_type == "logmel":
        return extract_logmel(wav_path, aug)
    if feature_type == "fft_full":
        return extract_fft_full(wav_path, aug)


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def extract_fft(wav_path, aug=None):
    y, fs = librosa.load(wav_path, sr=16000)
    y = pad(y)
    if aug != None:
        y = process_Rawboost_feature(y, algo=aug)
    return torch.Tensor(np.reshape(np.array(y), (-1, 64600)))

def extract_logmel(wav_path, aug=None, sample_rate=16000, n_mels=80, win_length=400, hop_length=160):
    
    waveform, sr = librosa.load(wav_path, sr=sample_rate)
    waveform = pad(waveform)
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=win_length,  # FFT size matches the window length
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=2.0  # Use power spectrogram
    )

    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    normalized_log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / np.std(log_mel_spec)
    tensor_input = torch.tensor(normalized_log_mel_spec, dtype=torch.float32).T 
    return tensor_input

def extract_fft_full(wav_path, aug=None):
    y, fs = librosa.load(wav_path, sr=16000)
    if aug != None:
        y = process_Rawboost_feature(y, algo=aug)

    return torch.Tensor(np.reshape(np.array(y), (1, -1)))


def main():
    r=extract_logmel('/netscratch/yelkheir/fft_test.wav')
    print(r.shape)
    return r

if __name__ == '__main__':
    main()
