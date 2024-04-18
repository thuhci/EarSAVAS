import os
import time
import hydra
import torch
import joblib
import numpy as np
import SAMoSA_data_utils.vggish_input as vggish_input

def SAMoSA_audio_data_prep(wave):
    return vggish_input.wavform_to_concat_examples(wave, 
                                                   lower_edge_hertz=10,
                                                   upper_edge_hertz=8000,
                                                   sr=16000)

@hydra.main(config_path="configs", config_name='config', version_base = '1.3')
def main(cfg):
    Dataset_config = cfg.Dataset

    seed = getattr(Dataset_config, 'seed', 0)
    np.random.seed(seed)
    
    audio_sr = getattr(Dataset_config, 'audio_sr', 16000)
    snippet_duration = Dataset_config.duration
    
    dataset_dir = Dataset_config.dataset_dir
    if not os.path.exists(dataset_dir):
        raise ValueError("The dataset directory does not exist, please run 1-prep_data.py and 2-mel_converter.py first")
    
    audio_dataset_path = f'ad_{audio_sr}_{snippet_duration}.pkl'
    print(audio_dataset_path)
    audio_dataset_path = os.path.join(dataset_dir, audio_dataset_path)

    print(audio_dataset_path)

    if not os.path.exists(audio_dataset_path):
        raise ValueError("The dataset directory does not exist, please run prep_data.py first")

    audio_data = joblib.load(audio_dataset_path)
    
    audio_data = {
        user: {
            label: {
                audio_file: [
                    SAMoSA_audio_data_prep(audio_data[user][label][audio_file][idx])
                    for idx in range(len(audio_data[user][label][audio_file]))
                ]
                for audio_file in audio_data[user][label]
            }
            for label in audio_data[user]
        }
        for user in audio_data
    }

    joblib.dump(audio_data, audio_dataset_path.replace('.pkl', '_samosa.pkl'))
    
if __name__ == '__main__':
    main()