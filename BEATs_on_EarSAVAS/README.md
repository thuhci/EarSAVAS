# BEATs on EarSAVAS Dataset

This submodule saved the code for the training and evaluation of BEATs on EarSAVAS. Now with the evaluation dataset, you can test the trained model on EarSAVAS. We will complete the training tutorial once the EarSAVAS paper is accepted by IMWUT 2024.

## Prerequisite

1. Create a conda environment with Python 3.8
2. Download some related system files with apt.

```
sudo apt-get update
sudo apt-get install -y ffmpeg build-essential
```

3. Download the poetry with pip

```
pip install poetry==1.3.2
```

4. Build the environment with poetry (run the command in the current directory)

```
poetry config virtualenvs.create false
poetry install --no-root
```

5. Prepare the dataset as illustrated in the Readme files in EarVAS and SAMoSA.

6. Change the audio and imu dataset path in test.py

```python
...
        audio_path = Absolute path of your audio dataset (e.g. xxx/ad_16000_1.pkl),
        imu_path = Absolute path of your imu dataset (e.g. xxx/id_100_1.pkl),
...
```

7. Download the pre-trained model from and save them somewhere. Change the model_dir in test.py to where you save the pre-trained models.
8. run the test.py and get the results in the current directory as follows

```
BEATs_on_EarSAVAS/
├── two_channel_audio_False.txt
├── feedback_audio_False.txt
└── feedforward_audio_False.txt
```
