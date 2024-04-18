import os
import time
import hydra
import torch
import joblib
import EarVAS_models as EarVAS_models
import numpy as np
import EarVAS_dataloaders as EarVAS_dataloaders
from EarVAS_traintest_utils import train, validate

print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

@hydra.main(config_path="configs", config_name='config', version_base = '1.3')
def main(cfg):
    Dataset_config = cfg.Dataset
    Model_config = cfg.Model

    seed = getattr(Dataset_config, 'seed', 0)
    np.random.seed(seed)
    
    audio_sr = getattr(Dataset_config, 'audio_sr', 16000)
    imu_sr = getattr(Dataset_config, 'imu_sr', 100)
    snippet_duration = Dataset_config.duration
    
    dataset_dir = Dataset_config.dataset_dir
    if not os.path.exists(dataset_dir):
        raise ValueError("The dataset directory does not exist, please run prep_data.py first")

    if Model_config.samosa:
        audio_dataset_path = f'ad_{audio_sr}_{snippet_duration}_samosa.pkl'
    else:
        audio_dataset_path = f'ad_{audio_sr}_{snippet_duration}_round2.pkl'

    if Model_config.samosa:
        imu_dataset_path = f'id_{imu_sr}_{snippet_duration}.pkl'
    else:
        imu_dataset_path = f'id_{imu_sr}_{snippet_duration}_round2.pkl'
    audio_dataset_path = os.path.join(dataset_dir, audio_dataset_path)
    imu_dataset_path = os.path.join(dataset_dir, imu_dataset_path)

    print(audio_dataset_path, imu_dataset_path)

    if not os.path.exists(audio_dataset_path) or \
        not os.path.exists(imu_dataset_path):
        raise ValueError("The dataset directory does not exist, please run prep_data.py first")

    audio_data = joblib.load(audio_dataset_path)
    imu_data = joblib.load(imu_dataset_path)

    num_mel_bins = getattr(Dataset_config, 'num_mel_bins', 128)
    target_length = getattr(Dataset_config, 'target_length', 128)
    freqm = getattr(Dataset_config, 'freqm', 0)
    timem = getattr(Dataset_config, 'timem', 0)

    audio_conf = {'num_mel_bins': num_mel_bins, 'target_length': target_length, 'freqm': freqm, 'timem': timem, 'mode': 'train'}

    training_user_list = Dataset_config.training_user_list
    validation_user_list = Dataset_config.validation_user_list
    testing_user_list = Dataset_config.testing_user_list
    
    raw_label_list = Dataset_config.label_list
    task = Model_config.task

    print(task)
    if task == 'SWITest_without_non_subjects':
        raw_label_list = [item for item in raw_label_list if 'non_subject' not in item]

    print(raw_label_list)
    raw_label_dict = {label: idx for idx, label in enumerate(raw_label_list)}

    print(training_user_list, len(training_user_list))
    print(validation_user_list, len(validation_user_list))
    print(testing_user_list, len(testing_user_list))
    print("Task: ", task)

    num_epochs = Model_config.num_epochs
    batch_size = Model_config.batch_size
    num_workers = Model_config.num_workers
    exp_dir = Model_config.exp_dir
    samosa = Model_config.samosa

    if task == 'SWITest_without_non_subjects' or task == 'SWITest_with_non_subjects':
        training_dataset = EarVAS_dataloaders.SWITestDataset(audio_data, training_user_list, raw_label_dict, task=task, audio_conf=audio_conf, specaug=True)
    else:
        training_dataset = EarVAS_dataloaders.EarSAVAS_Dataset(audio_data, imu_data, training_user_list, raw_label_dict, audio_conf=audio_conf, specaug=True, samosa=samosa)
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_audio_conf = {'num_mel_bins': num_mel_bins, 'target_length': target_length, 'mode': 'test'}
    if task == 'SWITest_without_non_subjects' or task == 'SWITest_with_non_subjects':
        validation_dataset = EarVAS_dataloaders.SWITestDataset(audio_data, validation_user_list, raw_label_dict, task=task, audio_conf=val_audio_conf)
        testing_dataset = EarVAS_dataloaders.SWITestDataset(audio_data, testing_user_list, raw_label_dict, task=task, audio_conf=val_audio_conf)
    else:
        validation_dataset = EarVAS_dataloaders.EarSAVAS_Dataset(audio_data, imu_data, validation_user_list, raw_label_dict, audio_conf=val_audio_conf, samosa=samosa)
        testing_dataset = EarVAS_dataloaders.EarSAVAS_Dataset(audio_data, imu_data, testing_user_list, raw_label_dict, audio_conf=val_audio_conf, samosa=samosa)

    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=True)

    label_list = training_dataset.label_list
    label_dict = training_dataset.label_dict
    print(label_list)
    print(label_dict)
    num_classes = len([item for item in label_list if 'non_subject' not in item])
    print(num_classes)

    feature_size = Model_config.single_modality_feature_size

    if task == 'two_channel_audio_and_imu':
        audio_model = EarVAS_models.EarVAS(num_classes, fusion=True, audio_channel='BiChannel', feature_size=feature_size, samosa=samosa)
    elif task == 'two_channel_audio':
        audio_model = EarVAS_models.EarVAS(num_classes, fusion=False, audio_channel='BiChannel', feature_size=feature_size, samosa=samosa)
    elif task == 'feedforward_audio':
        audio_model = EarVAS_models.EarVAS(num_classes, fusion=False, audio_channel='FeedForward', feature_size=feature_size, samosa=samosa)
    elif task == 'feedback_audio':
        audio_model = EarVAS_models.EarVAS(num_classes, fusion=False, audio_channel='FeedBack', feature_size=feature_size, samosa=samosa)
    elif task == 'imu_only':
        audio_model = EarVAS_models.EarVAS(num_classes, fusion=False, audio_channel='None', feature_size=feature_size, samosa=samosa)
    elif task == 'feedback_audio_and_imu':
        audio_model = EarVAS_models.EarVAS(num_classes, fusion=True, audio_channel='FeedBack', feature_size=feature_size, samosa=samosa)
    elif task == 'feedforward_audio_and_imu':
        audio_model = EarVAS_models.EarVAS(num_classes, fusion=True, audio_channel='FeedForward', feature_size=feature_size, samosa=samosa)
    elif task == 'SWITest_without_non_subjects' or task == 'SWITest_with_non_subjects':
        audio_model = EarVAS_models.EffNetMean(num_classes)
    else:
        raise ValueError('Model Unrecognized')

    print("\nCreating experiment directory: %s" % exp_dir)
    os.makedirs("%s/models" % exp_dir, exist_ok=True)
    joblib.dump(cfg, f"{exp_dir}/args_{task}.pkl")

    print('Now starting training for {:d} epochs'.format(num_epochs))
    train(audio_model, train_loader, val_loader, cfg)

    # test on the test set and sub-test set, model selected on the validation set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(exp_dir + f'/models/best_audio_model_{task}_SAMoSA_{samosa}.pth', map_location=device)
    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd)

    if task != 'SWITest_without_non_subjects':
        stats, _, macro_f1, confusion_matrix = validate(audio_model, test_loader, cfg, detail_analysis=True, label_list = label_list, label_dict = label_dict)
    else:
        stats, _, macro_f1, confusion_matrix = validate(audio_model, test_loader, cfg)

    test_acc = stats[0]['acc']
    print('---------------evaluate on the validation set---------------')
    print("Accuracy: {:.6f}".format(test_acc))
    print("test confusion matrix: ")
    print(confusion_matrix)

    with open(exp_dir + f'/confusion_matrix_{task}_SAMoSA_{samosa}.txt', 'a') as f:
        f.write(f'Validation Confusion Matrix:\n')
        f.write(str(confusion_matrix))
        f.write('\n')

if __name__ == '__main__':
    main()