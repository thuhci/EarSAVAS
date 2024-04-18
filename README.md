## The EarSAVAS Dataset: Enabling Subject-Aware Vocal Activity Sensing on Earables

### Introduction

---

EarSAVAS is a publicly available multi-modal dataset crafted for subject-aware human vocal activity sensing on earables, with 44.5 hours of synchronous audio and motion data collected from 42 participants, encompassing 8 different types of human vocal activities. Audio data consists of feed-forward and feedback microphones of active noise-cancelling earables with a sampling rate of 16kHz. IMU data consists of the 3-axis accelerometer data stream and the 3-axis data stream from the gyroscope, with a sampling rate of 100Hz.

Now only the evaluation dataset of EarSAVAS is available, consisting of the data from 8 users. We also release the pre-trained models we obtained to facilitate the evaluation of EarVAS series models. We will release the whole dataset once our dataset paper is accepted by the IMWUT 2024.

Since we only propose the evaluation dataset, researchers can conduct evaluations on our pre-trained EarVAS series models we released in this respiratory. 

### Download EarSAVAS

---

The EarSAVAS dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/earsavas/earsavas-dataset) now, with data from 8 users available. The description of the structure of our released dataset is listed below. For evaluation of our pre-trained model in EarSAVAS dataset paper, only download the cutted_data directory.

```
EarSAVAS_Dataset/
├── cutted_data/
│   │
│   ├── user_6_1    # All the data collected from the earables worn by one specific user
│   │   ├── audio/    # All the audio data of one specific user
│   │   │     ├── Cough		 # All the audio data of cough events originated from user_6_1
│   │   │     │     ├── user_6_1_1.wav 		# Audio data file, the numbers behind user_6_1_ hold no specific meaning
│   │   │     │     ├── user_6_1_3.wav
│   │   │     ├── Speech		# All the audio data of speech events originated from user_6_1
│   │   │     ├── Cough_non_subject		 # Cough events originated from user_6_2 but collected by earables of 6_1
│   │   │     ├── Speech_non_subject		# Speech events originated from user_6_2 but collected by earables of 6_1
│   │   │     ├── ...
│   │   └── imu/    # All the motion data of one specific user
│   │         ├── Cough		 # All the motion data of cough events originated from user_6_2
│   │         │     ├── user_6_1_1.pkl 		# IMU data file, the numbers is used to find the corresponding audio file
│   │         │     ├── user_6_1_3.pkl
│   │         ├── Speech		# All the motion data of speech events originated from user_6_1
│   │         ├── Cough_non_subject		 # All the motion data collected while user_6_2 coughs
│   │         ├── Speech_non_subject		# All the motion data collected while user_6_2 speech
│   │         ├── ...											
│   ├── user_15_1
│   ├── user_14_1  
│   ├── user_12_2  
│   ├── user_7_2  
│   ├── user_25_1  
│   ├── user_4_2  
│   └── user_3_1 
│
├── raw_data
├── annotation_files
├── split_channel_cutted_data
└── split_channel_raw_data
```



### Evaluate the pertained EarVAS/SAMoSA benchmark model locally

---

**Step 1.** Build a conda environment where the **Python version is 3.8**, then clone and download this repository and set it as the working directory, create a virtual environment, and install the dependencies.

```
cd EarSAVAS/
pip install -r requirements.txt 
```



**Step 2.** Download the EarSAVAS dataset and prepare data for EarVAS evaluation.

Download the data from [Kaggle](https://www.kaggle.com/datasets/earsavas/earsavas-dataset) and get the path of the dataset

```python
python3 prep_data.py Dataset.raw_data_dir=absolute_path_of_data Dataset.dataset_dir=absolute_path_where_you_want_to_keep_the_proposed_dataset
```



**Step 3.** Prepare the data for SAMoSA evaluation.

```
python3 SAMoSA_data_prepare.py Dataset.dataset_dir=the_dataset_dir_you_set_on_the_step2
```



**Step 4.** Download the pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1O0mGU9ziRWii0kGJFmhratkw9e6lsJry)



**Step 5.** Run the evaluation of our best models of EarVAS series models.

```
python3 EarVAS_evaluation.py Dataset.dataset_dir=absolute_path_of_where_you_store_dataset_files_in_step_2 Model.exp_dir=absolute_path_where_you_download_the_pretrained_models Model.task=$task Model.device=$device Model.samosa=True/False
```

The model task can be only selected from [two_channel_audio_and_imu, two_channel_audio, feedforward_audio, feedback_audio, imu_only, feedback_audio_and_imu, feedforward_audio_and_imu]

We use the samosa to select the evaluation on EarVAS or SAMoSA. For example, if the Model.task=two_channel_audio_and_imu and the Model.samosa=False, then you evaluate the performance of EarVAS with two-channel audio and motion data as input. However, if the Model.samosa=True, then you evaluate the performance of SAMoSA with the same input modalities.

The model device can be only selected from [cpu, cuda]. **We recommend evaluating our pre-trained models on the CPU. If the device is selected as cuda, then please ensure that the version of your CUDA is 12.2.** Otherwise, the results will differ from those reported in the original paper. 