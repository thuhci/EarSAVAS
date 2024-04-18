import torch
import joblib
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule

class EarSAVAS_Dataset(Dataset):
    def __init__(self, data, label_list, label_dict):
        self.final_data = data
        self.label_list = label_list
        self.label_dict = label_dict

    def __getitem__(self, index):

        mixed_data, label = self.final_data[index]
        audio_data, imu_data = mixed_data
        imu_data = imu_data.astype(np.float32)

        others_index = self.label_list.index('others')
        valid_label_length = others_index + 1

        label_indices = np.zeros(valid_label_length) + 0.00

        if label == 'others' or 'non_subject' in label:
            label_indices[self.label_dict['others']] = 1.0
        else:
            label_indices[self.label_dict[label]] = 1.0

        label_raw = self.label_dict[label]

        label_non_zero_index = np.nonzero(label_indices)[0][0]
        padding_mask = torch.zeros(2, audio_data.shape[1]).bool().squeeze(0)

        return audio_data, imu_data, padding_mask, label_non_zero_index, label_raw

    def __len__(self):
        return len(self.final_data)

def prep_EarSAVAS_data(audio_data, imu_data, user_list, label_list, training_subject_ratio=None):
    audio_data = {user: audio_data[user] for user in user_list}
    audio_data = {user: {label: audio_data[user][label] for label in audio_data[user] if label in label_list} for user in audio_data}
    imu_data = {user: imu_data[user] for user in user_list}
    imu_data = {user: {label: imu_data[user][label] for label in imu_data[user] if label in label_list} for user in imu_data}

    mixed_data = {
        user: {
            label: {
                audio_file: [
                    [audio_data[user][label][audio_file][idx], imu_data[user][label][audio_file.replace('audio', 'imu').replace('wav', 'pkl')][idx]]
                    for idx in range(len(audio_data[user][label][audio_file]))
                ]
                for audio_file in audio_data[user][label]
            }
            for label in audio_data[user]
        }
        for user in audio_data
    }

    for idx, user_name in enumerate(mixed_data):
        for label in mixed_data[user_name]:
            if training_subject_ratio is None or idx < len(mixed_data) * training_subject_ratio or label == 'others' or 'non_subject' in label:
                mixed_data[user_name][label] = [clip for audio_file in mixed_data[user_name][label] \
                                                    for clip in mixed_data[user_name][label][audio_file]]
            else:
                mixed_data[user_name][label] = []
            # mixed_data[user_name][label] = [clip for audio_file in mixed_data[user_name][label] \
                                                # for clip in mixed_data[user_name][label][audio_file]]
            
    final_data = []

    for user_name in mixed_data:
        for label in mixed_data[user_name]:
            for m_data in mixed_data[user_name][label]:
                if m_data[0].shape[1] != 0:
                    new_label = label.replace('Single_Cough', 'Cough')
                    new_label = new_label.replace('Continuous_Cough', 'Cough')
                    new_label = new_label.replace('Single_Cough_non_subject', 'Cough_non_subject')
                    new_label = new_label.replace('Continuous_Cough_non_subject', 'Cough_non_subject')
                    final_data.append([m_data, new_label])

    return final_data

class EarSAVASDataModule(LightningDataModule):
    def __init__(
        self,
        training_user_list: list,
        validation_user_list: list,
        testing_user_list: list,
        label_list: list,
        label_after_combine_list: list,
        audio_path: str = "ad_16000_1_round2.pkl",
        imu_path: str = "imu_100_1_round2.pkl",
        batch_size: int = 8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.training_user_list = training_user_list
        self.validation_user_list = validation_user_list
        self.testing_user_list = testing_user_list
        self.label_list = label_list
        self.label_after_combine_list = label_after_combine_list
        self.audio_path = audio_path
        self.imu_path = imu_path
        self.batch_size = batch_size

        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        audio_data = joblib.load(self.audio_path)
        imu_data = joblib.load(self.imu_path)

        self.label_dict = {label: idx for idx, label in enumerate(self.label_list)}

        self.training_final_data = prep_EarSAVAS_data(audio_data, imu_data, self.training_user_list, self.label_list)
        self.validation_final_data = prep_EarSAVAS_data(audio_data, imu_data, self.validation_user_list, self.label_list)
        self.testing_final_data = prep_EarSAVAS_data(audio_data, imu_data, self.testing_user_list, self.label_list)
        
        label_counter = Counter(data[1] for data in self.training_final_data)
        print(label_counter)

    def train_dataloader(self):
        self.label_dict = {label: idx for idx, label in enumerate(self.label_after_combine_list)}
        train_df = EarSAVAS_Dataset(
            data=self.training_final_data, label_list=self.label_after_combine_list, label_dict=self.label_dict
        )

        return DataLoader(train_df, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        self.label_dict = {label: idx for idx, label in enumerate(self.label_after_combine_list)}

        val_df = EarSAVAS_Dataset(
            data=self.validation_final_data, label_list=self.label_list, label_dict=self.label_dict
        )

        return DataLoader(val_df, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        self.label_dict = {label: idx for idx, label in enumerate(self.label_after_combine_list)}
        
        test_df = EarSAVAS_Dataset(
            data=self.testing_final_data, label_list=self.label_list, label_dict=self.label_dict
        )

        return DataLoader(test_df, batch_size=self.batch_size, shuffle=False)