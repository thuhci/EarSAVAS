# MFCC Spectrogram conversion code from VGGish, Google Inc.
# https://github.com/tensorflow/models/tree/master/research/audioset

import numpy as np
from scipy.io import wavfile
import SAMoSA_data_utils.mel_features as mel_features
import SAMoSA_data_utils.params as params

def wavfile_to_concat_examples(wav_file, lower_edge_hertz=params.MEL_MIN_HZ, 
                               upper_edge_hertz=params.MEL_MAX_HZ):
    sr, wav_data = wavfile.read(wav_file)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    data = wav_data / 32768.0    # Convert to [-1.0, +1.0]

    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(data,
                                               audio_sample_rate=sr,
                                               log_offset=params.LOG_OFFSET,
                                               window_length_secs=params.STFT_WINDOW_LENGTH_SECONDS,
                                               hop_length_secs=params.STFT_HOP_LENGTH_SECONDS,
                                               num_mel_bins=params.NUM_MEL_BINS,
                                               lower_edge_hertz=lower_edge_hertz,
                                               upper_edge_hertz=upper_edge_hertz)

    return log_mel

def wavform_to_concat_examples(wav_data, lower_edge_hertz=params.MEL_MIN_HZ, 
                               upper_edge_hertz=params.MEL_MAX_HZ, sr=16000):
    # assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    # data = wav_data / 32768.0    # Convert to [-1.0, +1.0]

    # Convert to mono.
    # if len(data.shape) > 1:
        # data = np.mean(data, axis=1)
    assert len(wav_data.shape) == 2, 'Bad sample shape: %r' % wav_data.shape

    # Compute log mel spectrogram features.
    log_mel_1 = mel_features.log_mel_spectrogram(wav_data[0, :],
                                               audio_sample_rate=sr,
                                               log_offset=params.LOG_OFFSET,
                                               window_length_secs=params.STFT_WINDOW_LENGTH_SECONDS,
                                               hop_length_secs=params.STFT_HOP_LENGTH_SECONDS,
                                               num_mel_bins=params.NUM_MEL_BINS,
                                               lower_edge_hertz=lower_edge_hertz,
                                               upper_edge_hertz=upper_edge_hertz)

    log_mel_2 = mel_features.log_mel_spectrogram(wav_data[1, :],
                                                  audio_sample_rate=sr,
                                                  log_offset=params.LOG_OFFSET,
                                                  window_length_secs=params.STFT_WINDOW_LENGTH_SECONDS,
                                                  hop_length_secs=params.STFT_HOP_LENGTH_SECONDS,
                                                  num_mel_bins=params.NUM_MEL_BINS,
                                                  lower_edge_hertz=lower_edge_hertz,
                                                  upper_edge_hertz=upper_edge_hertz)
    
    log_mel = np.stack((log_mel_1, log_mel_2), axis=0)
    return log_mel
