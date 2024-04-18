import os
from pytorch_lightning import Trainer
from datamodules.ECS50DataModule import EarSAVASDataModule
from fine_tune.transferLearning import BEATsEarVASTransferLearningModel
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

model_dir = './lightning_logs'
version_files = [os.path.join(model_dir, f'version_{idx}/checkpoints') for idx in range(21, 24)]
version_file_paths = [os.path.join(item, file) for item in version_files for file in os.listdir(item)]

for version_files in version_file_paths:
    model = BEATsEarVASTransferLearningModel.load_from_checkpoint(version_files)
    data_module = EarSAVASDataModule(
        training_user_list=[],
        validation_user_list=[],
        testing_user_list = ['user_6_1', 'user_15_1', 'user_14_1', 'user_12_2', 'user_7_2', 'user_25_1', 'user_4_2', 'user_3_1'],
        label_list = ['Blow_Nose', 'Throat_Clear', 'Sniff', 'Sigh', 'Drink', 'Speech', 'Single_Cough', 'Continuous_Cough', 'Chewing', 'others', 'Blow_Nose_non_subject', 'Throat_Clear_non_subject', 'Sniff_non_subject', 'Sigh_non_subject', 'Drink_non_subject', 'Speech_non_subject', 'Single_Cough_non_subject', 'Continuous_Cough_non_subject', 'Chewing_non_subject'],
        label_after_combine_list = ['Blow_Nose', 'Throat_Clear', 'Sniff', 'Sigh', 'Drink', 'Speech', 'Cough', 'Chewing', 'others', 'Blow_Nose_non_subject', 'Throat_Clear_non_subject', 'Sniff_non_subject', 'Sigh_non_subject', 'Drink_non_subject', 'Speech_non_subject', 'Cough_non_subject', 'Chewing_non_subject'],
        audio_path = 'ad_16000_1.pkl',
        imu_path = 'id_100_1.pkl',
        batch_size = 32
    )
    data_module.setup('test')
    test_dataloader = data_module.test_dataloader()

    trainer = Trainer(gpus = [1], accelerator = 'gpu')
    trainer.test(model, dataloaders=test_dataloader)