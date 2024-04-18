# import libraries
import os
import hydra
import numpy as np
import librosa
import joblib
from tqdm import tqdm

# load the data of given user with given label
# audio_data: {audio_file_path: audio_data}
# imu_data: {imu_file_path: imu_data}
def load_user_label(user_name, label, audio_sr, raw_dir):
    if not os.path.exists(os.path.join(raw_dir, user_name, 'audio', label)):
        print(os.path.join(raw_dir, user_name, 'audio', label))
        raise Exception('The user does not have the label.')
    
    audio_files = os.listdir(os.path.join(raw_dir, user_name, 'audio', label))

    audio_data, imu_data = {}, {}

    for audio_file in audio_files:
        if audio_file.endswith('.wav'):
            audio_file_path = os.path.join(raw_dir, user_name, 'audio', label, audio_file)
            imu_file = audio_file.replace('audio', 'imu')
            imu_file_path = os.path.join(raw_dir, user_name, 'imu', label, imu_file)
            imu_file_path = imu_file_path.replace('.wav', '.pkl')
            audio_data[audio_file_path], _ = librosa.load(os.path.join(raw_dir, user_name, 'audio', label, audio_file), sr = audio_sr, mono=False)
            imu_data[imu_file_path] = joblib.load(imu_file_path)
            imu_data[imu_file_path] = imu_data[imu_file_path].T

    return audio_data, imu_data

# audio_data: {label: {audio_file_path: audio_data}}
# imu_data: {label: {imu_file_path: imu_data}}
def load_user(user_name, audio_sr, raw_dir):
    if not os.path.exists(os.path.join(raw_dir, user_name)):
        raise Exception('The user does not exist.')
    
    audio_labels = os.listdir(os.path.join(raw_dir, user_name, 'audio'))

    audio_data, imu_data = {}, {}

    for label in audio_labels:
        if os.path.isdir(os.path.join(raw_dir, user_name, 'audio', label)):
            audio_data_label, imu_data_label = load_user_label(user_name, label, audio_sr, raw_dir)
            real_label = label.replace('Read', 'Speech').replace('Eat', 'Chewing')
            audio_data[real_label] = audio_data_label
            imu_data[real_label] = imu_data_label

    return audio_data, imu_data

# load all the users
# audio_data: {user_name: {label: {audio_file_path: audio_data}}}
# imu_data: {user_name: {label: {imu_file_path: imu_data}}}
def load_all_users(audio_sr, imu_sr, duration, user_list, raw_dir):
    audio_data, imu_data = {}, {}
    for user in tqdm(user_list, desc = 'Load all users'):
        if os.path.isdir(os.path.join(raw_dir, user)):
            print("Prepare Data For: ", user)
            audio_data_user, imu_data_user = load_user(user, audio_sr, raw_dir)
            labels = list(audio_data_user.keys())
            # go through each label
            for label in labels:
                # get the list of the audio files
                audio_files = list(audio_data_user[label].keys())
                # go through each audio file
                for audio_file in audio_files:
                    # get the audio data
                    audio_data_file = audio_data_user[label][audio_file]
                    # get the imu data
                    audio_root_dir_name = '/'.join(audio_file.split('/')[:-3])
                    audio_sub_dir_name = '/'.join(audio_file.split('/')[-2:])
                    imu_file = os.path.join(audio_root_dir_name, 'imu', audio_sub_dir_name)
                    imu_file = imu_file.replace('.wav', '.pkl')
                    imu_data_file = imu_data_user[label][imu_file]
                    
                    # cut the file into several snippets with given duration and pad the final snippet
                    # Note that the audio_sr and imu_sr are different
                    audio_data_snippet = []
                    imu_data_snippet = []
                    # get the number of snippets
                    num_snippets = int(imu_data_file.shape[1] / (imu_sr * duration))
                    num_snippets = int(audio_data_file.shape[1] / (audio_sr * duration))
                    # go through each snippet
                    for i in range(num_snippets):
                        audio_start_index = int(i * audio_sr * duration)
                        audio_end_index = int((i + 1) * audio_sr * duration)
                        imu_start_index = int(i * imu_sr * duration)
                        imu_end_index = int((i + 1) * imu_sr * duration)
                        audio_data_snippet.append(audio_data_file[:, audio_start_index : audio_end_index])
                        imu_data_snippet.append(imu_data_file[:, imu_start_index : imu_end_index])
                    
                    # pad the final snippet
                    final_audio_data = audio_data_file[:, int((num_snippets) * audio_sr * duration) : ]
                    final_imu_data = imu_data_file[:, int((num_snippets) * imu_sr * duration) : ]
                    # get the number of samples to pad
                    audio_num_samples_to_pad = int(audio_sr * duration) - final_audio_data.shape[1]
                    imu_num_samples_to_pad = int(imu_sr * duration) - final_imu_data.shape[1]

                    # pad the tail of the audio data
                    if num_snippets == 0:
                        final_audio_data = np.pad(final_audio_data, ((0, 0), (0, audio_num_samples_to_pad)), 'constant', constant_values = (0, 0))
                        final_imu_data = np.pad(final_imu_data, ((0, 0), (0, imu_num_samples_to_pad)), 'constant', constant_values = (0, 0))

                        audio_data_snippet.append(final_audio_data)
                        imu_data_snippet.append(final_imu_data)

                    audio_data_user[label][audio_file] = audio_data_snippet
                    imu_data_user[label][imu_file] = imu_data_snippet
            audio_data[user] = audio_data_user
            imu_data[user] = imu_data_user

    return audio_data, imu_data

# target_dir: the directory to store the final pkl files
# user_list: the list of the users to load
# raw_dir: the directory of the raw data
def prepare_dataset(audio_sr, imu_sr, duration, target_dir, user_list, raw_dir):
    os.makedirs(target_dir, exist_ok = True)
    audio_data, imu_data = load_all_users(audio_sr, imu_sr, duration, user_list, raw_dir)
    
    audio_save_file_path = f'ad_{audio_sr}_{duration}.pkl'
    imu_save_file_path = f'id_{imu_sr}_{duration}.pkl'
    
    audio_save_file_path = os.path.join(target_dir, audio_save_file_path)
    imu_save_file_path = os.path.join(target_dir, imu_save_file_path)

    joblib.dump(audio_data, audio_save_file_path)
    joblib.dump(imu_data, imu_save_file_path)

@hydra.main(config_path='configs', config_name='config', version_base = '1.3')
def main(cfg):
    Dataset_config = cfg.Dataset

    audio_sr = getattr(Dataset_config, 'audio_sr', 16000)
    imu_sr = getattr(Dataset_config, 'imu_sr', 100)
    snippet_duration = Dataset_config.duration

    raw_data_dir = Dataset_config.raw_data_dir
    target_dir = Dataset_config.dataset_dir
    training_user_list = Dataset_config.training_user_list
    validation_user_list = Dataset_config.validation_user_list
    # training_user_list = []
    # validation_user_list = []
    testing_user_list = Dataset_config.testing_user_list
    user_list = training_user_list + validation_user_list + testing_user_list

    prepare_dataset(audio_sr, imu_sr, snippet_duration, target_dir, user_list, raw_data_dir)

if __name__ == "__main__":
    main()