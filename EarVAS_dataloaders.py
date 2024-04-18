import torch
import torchaudio
import numpy as np
import torch.nn.functional
from collections import Counter
from torch.utils.data import Dataset

class EarSAVAS_Dataset(Dataset):
    def __init__(self, audio_data, imu_data, user_list, label_dict, audio_conf=None, specaug=False, samosa=False):
        self.user_list = user_list
        self.label_dict = label_dict
        self.samosa = samosa

        self.audio_data = {user: audio_data[user] for user in self.user_list}
        self.audio_data = {user: {label: self.audio_data[user][label] for label in self.audio_data[user] if label in label_dict.keys() or label == 'Cough'} for user in self.audio_data}
        self.imu_data = {user: imu_data[user] for user in self.user_list}
        self.imu_data = {user: {label: self.imu_data[user][label] for label in self.imu_data[user] if label in label_dict.keys() or label == 'Cough'} for user in self.imu_data}

        self.mixed_data = {
            user: {
                label: {
                    audio_file: [
                        [self.audio_data[user][label][audio_file][idx], self.imu_data[user][label][audio_file.replace('audio', 'imu').replace('wav', 'pkl')][idx]]
                        for idx in range(len(self.audio_data[user][label][audio_file]))
                    ]
                    for audio_file in self.audio_data[user][label]
                }
                for label in self.audio_data[user]
            }
            for user in self.audio_data
        }

        for user_name in self.mixed_data:
            for label in self.mixed_data[user_name]:
                self.mixed_data[user_name][label] = [clip for audio_file in self.mixed_data[user_name][label] \
                                                    for clip in self.mixed_data[user_name][label][audio_file]]

        self.final_data = []
        
        for user_name in self.mixed_data:
            for label in self.mixed_data[user_name]:
                for mixed_data in self.mixed_data[user_name][label]:
                    if mixed_data[0].shape[1] != 0:
                        new_label = label.replace('Single_Cough', 'Cough')
                        new_label = new_label.replace('Continuous_Cough', 'Cough')
                        new_label = new_label.replace('Single_Cough_non_subject', 'Cough_non_subject')
                        new_label = new_label.replace('Continuous_Cough_non_subject', 'Cough_non_subject')
                        self.final_data.append([mixed_data, new_label])

        self.label_list = list(self.label_dict.keys())
        Single_Cough_index = self.label_list.index('Single_Cough')
        self.label_list.insert(Single_Cough_index, 'Cough')
        Single_Cough_non_subject_index = self.label_list.index('Single_Cough_non_subject')
        self.label_list.insert(Single_Cough_non_subject_index, 'Cough_non_subject')
        self.label_list.remove('Single_Cough')
        self.label_list.remove('Single_Cough_non_subject')
        self.label_list.remove('Continuous_Cough')
        self.label_list.remove('Continuous_Cough_non_subject')
        self.label_dict = {label: idx for idx, label in enumerate(self.label_list)}

        label_counter = Counter(data[1] for data in self.final_data)
        print(label_counter.most_common())

        self.audio_conf = audio_conf
        self.mode = self.audio_conf.get('mode')
        self.melbins = self.audio_conf.get('num_mel_bins')

        if specaug == True:
            self.freqm = self.audio_conf.get('freqm')
            self.timem = self.audio_conf.get('timem')

        self.specaug = specaug

    def _wav2fbank(self, waveform, sr):
        waveform = torch.tensor(waveform, dtype = torch.float32)      
        mean_vals = torch.mean(waveform, axis=1, keepdims=True)
        waveform = waveform - mean_vals

        waveform_1 = torch.unsqueeze(waveform[0, :], dim=0)
        waveform_2 = torch.unsqueeze(waveform[1, :], dim=0)

        fbank1 = torchaudio.compliance.kaldi.fbank(waveform_1, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                    window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        fbank2 = torchaudio.compliance.kaldi.fbank(waveform_2, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                    window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.audio_conf.get('target_length', 1056)
        n_frames = fbank1.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank1 = m(fbank1)
            fbank2 = m(fbank2)
        elif p < 0:
            fbank1 = fbank1[0:target_length, :]
            fbank2 = fbank2[0:target_length, :]
        
        return fbank1, fbank2

    def __getitem__(self, index):

        mixed_data, label = self.final_data[index]
        audio_data, imu_data = mixed_data
        imu_data = imu_data.astype(np.float32)

        others_index = self.label_list.index('others')
        valid_label_length = others_index + 1

        label_indices = np.zeros(valid_label_length) + 0.00
        label_raw = self.label_dict[label]

        if label == 'others' or 'non_subject' in label:
            label_indices[self.label_dict['others']] = 1.0
        else:
            label_indices[self.label_dict[label]] = 1.0
            
        if not self.samosa:
            fbank1, fbank2 = self._wav2fbank(audio_data, 16000)
            label_indices = torch.FloatTensor(label_indices)

            if self.specaug == True:
                freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
                timem = torchaudio.transforms.TimeMasking(self.timem)
                fbank1 = torch.transpose(fbank1, 0, 1)
                fbank1 = fbank1.unsqueeze(0)
                fbank1 = freqm(fbank1)
                fbank1 = timem(fbank1)
                fbank1 = fbank1.squeeze(0)
                fbank1 = torch.transpose(fbank1, 0, 1)

                fbank2 = torch.transpose(fbank2, 0, 1)
                fbank2 = fbank2.unsqueeze(0)
                fbank2 = freqm(fbank2)
                fbank2 = timem(fbank2)
                fbank2 = fbank2.squeeze(0)
                fbank2 = torch.transpose(fbank2, 0, 1)

            fbank = torch.stack((fbank1, fbank2), dim=0)
            fbank = (fbank + 3.05) / 5.42

            if self.mode == 'train':
                fbank = torch.roll(fbank, np.random.randint(0, 1024), 0)
        else:
            fbank = audio_data

        return fbank, imu_data, label_indices, label_raw

    def __len__(self):
        return len(self.final_data)

class SWITestDataset(Dataset):
    def __init__(self, audio_data, user_list, label_dict, task, audio_conf=None, specaug=False):
        self.task = task

        self.user_list = user_list
        self.label_dict = label_dict

        self.audio_data = {user: audio_data[user] for user in self.user_list}
        self.audio_data = {user: {label: self.audio_data[user][label] for label in self.audio_data[user] if label in label_dict.keys() or label == 'Cough'} for user in self.audio_data}

        for user_name in self.audio_data:
            for label in self.audio_data[user_name]:
                self.audio_data[user_name][label] = [clip for audio_file in self.audio_data[user_name][label] \
                                                    for clip in self.audio_data[user_name][label][audio_file]]
        self.final_data = []
        for user_name in self.audio_data:
            for label in self.audio_data[user_name]:
                for mixed_data in self.audio_data[user_name][label]:
                    try:
                        if mixed_data.shape[1] != 0:
                            new_label = label.replace('Single_Cough', 'Cough')
                            new_label = new_label.replace('Continuous_Cough', 'Cough')
                            if task == 'SWITest_with_non_subjects':
                                new_label = new_label.replace('Single_Cough_non_subject', 'Cough_non_subject')
                                new_label = new_label.replace('Continuous_Cough_non_subject', 'Cough_non_subject')
                            self.final_data.append([mixed_data, new_label])
                    except:
                        print(user_name, label, mixed_data.shape)

        self.label_list = list(self.label_dict.keys())
        Single_Cough_index = self.label_list.index('Single_Cough')
        self.label_list.insert(Single_Cough_index, 'Cough')
        self.label_list.remove('Single_Cough')
        self.label_list.remove('Continuous_Cough')

        if task == 'SWITest_with_non_subjects':
            Single_Cough_non_subject_index = self.label_list.index('Single_Cough_non_subject')
            self.label_list.insert(Single_Cough_non_subject_index, 'Cough_non_subject')        
            self.label_list.remove('Single_Cough_non_subject')
            self.label_list.remove('Continuous_Cough_non_subject')
        
        self.label_dict = {label: idx for idx, label in enumerate(self.label_list)}

        label_counter = Counter(data[1] for data in self.final_data)
        print(label_counter.most_common())

        self.audio_conf = audio_conf
        self.mode = self.audio_conf.get('mode')
        self.melbins = self.audio_conf.get('num_mel_bins')

        if specaug == True:
            self.freqm = self.audio_conf.get('freqm')
            self.timem = self.audio_conf.get('timem')

        self.specaug = specaug

    def _wav2fbank(self, waveform, sr):
        waveform = torch.tensor(waveform, dtype = torch.float32)        
        # extract the feed-forward microphone audio
        waveform = waveform[1]
        waveform = waveform.unsqueeze(0)
        assert waveform.shape[0] == 1
        waveform = waveform - waveform.mean()
        
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        target_length = self.audio_conf.get('target_length', 1056)
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def __getitem__(self, index):

        audio_data, label = self.final_data[index]

        if self.task == 'SWITest_without_non_subjects':
            label_indices = np.zeros(len(self.label_dict)) + 0.00
            label_indices[self.label_dict[label]] = 1.0
        elif self.task == 'SWITest_with_non_subjects':
            others_index = self.label_list.index('others')
            valid_label_length = others_index + 1

            label_indices = np.zeros(valid_label_length) + 0.00
            label_raw = self.label_dict[label]

            if label == 'others' or 'non_subject' in label:
                label_indices[self.label_dict['others']] = 1.0
            else:
                label_indices[self.label_dict[label]] = 1.0
        else:
            raise ValueError('task not supported, please check the task name')

        fbank = self._wav2fbank(audio_data, 16000)    
        label_indices = torch.FloatTensor(label_indices)

        if self.specaug == True:
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            timem = torchaudio.transforms.TimeMasking(self.timem)
            fbank = torch.transpose(fbank, 0, 1)
            fbank = fbank.unsqueeze(0)
            fbank = freqm(fbank)
            fbank = timem(fbank)
            fbank = fbank.squeeze(0)
            fbank = torch.transpose(fbank, 0, 1)

        # mean/std is get from the val set as a prior.
        fbank = (fbank + 3.05) / 5.42

        if self.mode == 'train':
            fbank = torch.roll(fbank, np.random.randint(0, 1024), 0)

        if self.task == 'SWITest_without_non_subjects':
            return fbank, label_indices
        elif self.task == 'SWITest_with_non_subjects':
            return fbank, label_indices, label_raw
        else:
            raise ValueError('task not supported, please check the task name')

    def __len__(self):
        return len(self.final_data)