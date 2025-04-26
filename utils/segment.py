import torch
import numpy as np
from ssqueezepy import cwt, Wavelet
from utils.wavelet import wavelet_bandpass
from config.config import Config


def segment_signal(transformed_signals, segment_length=Config.segment_length_IMR, overlap=Config.overlap):
    segmented_signals = []
    step_size = int(segment_length * (1 - overlap))

    for signal in transformed_signals:
        segments = []
        for start in range(0, len(signal) - segment_length + 1, step_size):
            segment = signal[start:start + segment_length]
            segments.append(segment)
        segmented_signals.append(np.array(segments))

    return segmented_signals

def preprocess_signals(signals, segment_length=Config.segment_length, overlap=Config.overlap):
    processed_signals = []
    for signal in signals:
        transformed_signals = wavelet_bandpass(signal)
        segmented_signals = segment_signal(transformed_signals, segment_length=segment_length, overlap=overlap)
        processed_signals.append(segmented_signals)
    return processed_signals
