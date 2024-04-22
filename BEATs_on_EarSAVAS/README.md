# Train and Evaluate BEATs on EarSAVAS Dataset

This submodule saved the code for the training and evaluation of BEATs on EarSAVAS. 

### Prerequisite

---

1. Create a conda environment with Python 3.8
2. Download related system files with apt.

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

5. Add the current directory into the PYTHONPATH

```
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

6. Prepare the dataset as illustrated in the Readme file for EarVAS and SAMoSA. (Readme file in the parent directory)

### Training BEATs on EarSAVAS Dataset

---

1. Download the `BEATs_iter3_plus_AS2M.pt` model from [Google Drive](https://drive.google.com/drive/folders/15LMWkKMJ_zxY9pXaQxNoI9yfNeTpw_l8).
2. Run the following poetry command under the current directory.

```
poetry run fine_tune/trainer.py fit
		-c fine_tune/config.yaml
		--model.model_path absolute_path_of_where_you_store_BEATs_iter3_plus_AS2M.pt
		--model.task $task
		--data.audio_path absolute_path_of_ad_{sample_rate}_{duration}.pkl (after you run the prep_data.py)
		--data.imu_path absolute_path_of_id_{sample_rate}_{duration}.pkl (after you run the prep_data.py)
```

The model task can be only selected from [two_channel_audio, feedforward_audio, feedback_audio]

### Evaluate our pre-trained BEATs on EarSAVAS Dataset

---

1. Download the three version folder from [Google Drive](https://drive.google.com/drive/folders/15LMWkKMJ_zxY9pXaQxNoI9yfNeTpw_l8).
2. Change the model_dir in test.py to the place you save the three directory

3. Change the audio and imu dataset path in test.py

```python
...
        audio_path = Absolute path of your audio dataset (e.g. xxx/ad_16000_1.pkl),
        imu_path = Absolute path of your imu dataset (e.g. xxx/id_100_1.pkl),
...
```

4. run the test.py (python3 test.py) and get the results in the current directory as follows

```
BEATs_on_EarSAVAS/
├── two_channel_audio_False.txt
├── feedback_audio_False.txt
└── feedforward_audio_False.txt
```
