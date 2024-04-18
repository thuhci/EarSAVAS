import numpy as np
import sys
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torchmetrics import Accuracy
from pycm import *
from io import StringIO

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from BEATs.BEATs import BEATs, BEATsConfig, IMUFeatureExtractor, FocalLossMulti

class BEATsEarVASTransferLearningModel(pl.LightningModule):
    def __init__(
        self,
        num_target_classes: int = 9,
        milestones: int = 5,
        batch_size: int = 32,
        lr: float = 1e-3,
        lr_scheduler_gamma: float = 1e-1,
        num_workers: int = 6,
        task: str = "two_channel_audio_and_imu",
        modality_feature_dim: int = 256,
        model_path: str = "/model/BEATs_iter3_plus_AS2M.pt",
        ft_entire_network: bool = False, # Boolean on whether the classifier layer + BEATs should be fine-tuned
        **kwargs,
    ) -> None:
        """TransferLearningModel.
        Args:
            lr: Initial learning rate
        """
        super().__init__()
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.milestones = milestones
        self.num_target_classes = num_target_classes
        self.task = task
        print(self.task)
        self.modality_feature_dim = modality_feature_dim
        self.ft_entire_network_beats = ft_entire_network
        self.label_list = ['Blow_Nose', 'Throat_Clear', 'Sniff', 'Sigh', 'Drink', 'Speech', 'Cough', 'Chewing', 'others', 'Blow_Nose_non_subject', 'Throat_Clear_non_subject', 'Sniff_non_subject', 'Sigh_non_subject', 'Drink_non_subject', 'Speech_non_subject', 'Cough_non_subject', 'Chewing_non_subject']
        self.label_dict = {label: idx for idx, label in enumerate(self.label_list)}

        self.fp_label_dict = {label: 0 for label in self.label_list}
        self.fn_label_dict = {label: 0 for label in self.label_list}

        self.false_multi_dict = {label: {} for label in self.label_list if label != 'others' and 'non_subject' not in label}
        for label in self.false_multi_dict:
            self.false_multi_dict[label] = {label: 0 for label in self.label_list}

        self.valid_label_idx = [idx for idx, label in enumerate(self.label_list) if label != 'others' and 'non_subject' not in label]
        self.others_idx = self.label_list.index('others') 


        # Initialise BEATs model
        self.checkpoint = torch.load(model_path)
        self.cfg = BEATsConfig(
            {
                **self.checkpoint["cfg"],
                "predictor_class": self.num_target_classes,
                "finetuned_model": False,
            }
        )


        self._build_model()

        self.train_acc = Accuracy(
            task="multiclass", num_classes=self.num_target_classes
        )
        self.valid_acc = Accuracy(
            task="multiclass", num_classes=self.num_target_classes
        )
        self.save_hyperparameters()

    def _build_model(self):
        # 1. Load the pre-trained network
        self.beats = BEATs(self.cfg)
        self.beats.load_state_dict(self.checkpoint["model"])

        if 'imu' in self.task:
            self.imu_feature_extractor = IMUFeatureExtractor(self.modality_feature_dim)
        
        if 'two_channel' in self.task:
            self.beats.patch_embedding = nn.Conv2d(2, self.cfg.embed_dim, kernel_size=self.cfg.input_patch_size, stride=self.cfg.input_patch_size,
                                            bias=self.cfg.conv_bias)
            self.beats.input_patch_size = self.cfg.input_patch_size
        

        # 2. Classifier
        self.beats_feature_fc = nn.Linear(self.cfg.encoder_embed_dim, self.modality_feature_dim)
        # self.beats_feature_fc = nn.Linear(self.cfg.encoder_embed_dim, self.cfg.predictor_class)
        if 'imu' in self.task and 'two_channel' in self.task:        
            self.fc = nn.Linear(2 * self.modality_feature_dim, self.cfg.predictor_class)
        else:
            self.fc = nn.Linear(self.modality_feature_dim, self.cfg.predictor_class)

    def extract_features(self, x, padding_mask=None):
        if padding_mask != None:
            x, _ = self.beats.extract_features(x, padding_mask)
        else:
            x, _ = self.beats.extract_features(x)
        return x

    def forward(self, audio, imu, padding_mask=None):
        """Forward pass. Return x"""

        if 'feedback' in self.task:
            audio = audio[:, 0, :]
        elif 'feedforward' in self.task:
            audio = audio[:, 1, :]

        # if padding_mask != None:
        #     x, _ = self.beats.extract_features(audio, padding_mask)
        # else:
        #     x, _ = self.beats.extract_features(audio)

        # res = audio_features.mean(dim=1)
        if self.task != 'imu_only':
            # Get the representation
            if padding_mask != None:
                x, _ = self.beats.extract_features(audio, padding_mask)
            else:
                x, _ = self.beats.extract_features(audio)

            #     # Get the logits
            audio_features = self.beats_feature_fc(x)
            audio_features = audio_features.mean(dim=1)
        
        
        if 'imu' in self.task:
            imu_features = self.imu_feature_extractor(imu)
        
        if 'imu' in self.task and 'two_channel' in self.task:
            x = torch.cat((audio_features, imu_features), dim=1)
        elif self.task != 'imu_only':
            x = audio_features
        else:
            x = imu_features
        res = self.fc(x)
        # x = self.fc(x)

        # Mean pool the second layer
        # return x
        return res

    def loss(self, lprobs, labels):
        # self.loss_func = nn.CrossEntropyLoss()
        self.loss_func = FocalLossMulti(alpha = [0.35360856584308326, 0.07880689123104302, 0.13360323641582555, 
                                            0.0894736825693862, 0.1682411125236322, 0.02513590968890248, 
                                            0.10366172702597115, 0.04619501733699732, 0.0012738573651589443], gamma=2)
        return self.loss_func(lprobs, labels)

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        audio, imu, padding_mask, y_true, label_raw = batch
        y_probs = self.forward(audio, imu)

        # 2. Compute loss
        train_loss = self.loss(y_probs, y_true)

        # 3. Compute accuracy:
        self.log("train_acc", self.train_acc(y_probs, y_true), prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        audio, imu, padding_mask, y_true, label_raw = batch
        y_probs = self.forward(audio, imu)
        y_preds = torch.argmax(y_probs, dim=1)

        # 2. Compute loss
        self.log("val_loss", self.loss(y_probs, y_true), prog_bar=True)

        # 3. Compute accuracy:
        self.log("val_acc", self.valid_acc(y_probs, y_true), prog_bar=True)

        return {'preds': y_preds, 'targets': y_true}
    
    def validation_epoch_end(self, outputs):
        all_preds = torch.cat([x['preds'] for x in outputs], dim=0)
        all_labels = torch.cat([x['targets'] for x in outputs], dim=0)
        cm = ConfusionMatrix(actual_vector=all_labels.cpu().numpy(), predict_vector=all_preds.cpu().numpy())

        print('*' * 10 + "Validation Process" + '*' * 10)
        cm.print_matrix()
        cm.stat(summary=True)
        self.log("val_f1_macro", cm.F1_Macro, prog_bar=True)
        print('*' * 10 + "Validation Process" + '*' * 10)

    def test_step(self, batch, batch_idx):
        # 1. Forward pass:
        audio, imu, padding_mask, y_true, label_raw = batch
        y_probs = self.forward(audio, imu)
        y_preds = torch.argmax(y_probs, dim=1)

        index_predictions = y_preds.cpu().numpy()
        raw_labels = y_true.cpu().numpy()
        label_raw = label_raw.cpu().numpy()
        fp_indexes = np.where((label_raw > 7) & (index_predictions <= 7))[0]
        fn_indexes = np.where((label_raw <= 7) & (index_predictions > 7))[0]
        
        raw_raw_labels = label_raw
        fp_raw_labels = raw_raw_labels[fp_indexes]
        fn_raw_labels = raw_raw_labels[fn_indexes]
        label_dict_reverse = {idx: label for label, idx in self.label_dict.items()}
        fp_raw_labels = [label_dict_reverse[label] for label in fp_raw_labels]
        fn_raw_labels = [label_dict_reverse[label] for label in fn_raw_labels]

        for label in self.label_list:
            self.fp_label_dict[label] += fp_raw_labels.count(label)
            self.fn_label_dict[label] += fn_raw_labels.count(label)

        for valid_idx in self.valid_label_idx:
            valid_label = label_dict_reverse[valid_idx]
            fp_indexes = np.where(((raw_labels == self.others_idx) & (index_predictions == valid_idx)))[0]
            fp_raw_labels = raw_raw_labels[fp_indexes]
            fp_raw_labels = [label_dict_reverse[label] for label in fp_raw_labels]
            for label in self.label_list:
                self.false_multi_dict[valid_label][label] += fp_raw_labels.count(label)


        # 2. Compute loss
        self.log("test_loss", self.loss(y_probs, y_true), prog_bar=True)

        # 3. Compute accuracy:
        self.log("test_acc", self.valid_acc(y_probs, y_true), prog_bar=True)

        return {'preds': y_preds, 'targets': y_true}
    
    def test_epoch_end(self, outputs):
        captured_output = StringIO()          # Create StringIO object
        sys.stdout = captured_output          #  and redirect stdout.
        
        all_preds = torch.cat([x['preds'] for x in outputs], dim=0)
        all_labels = torch.cat([x['targets'] for x in outputs], dim=0)
        cm = ConfusionMatrix(actual_vector=all_labels.cpu().numpy(), predict_vector=all_preds.cpu().numpy())

        print('*' * 10 + "Testing Process" + '*' * 10)
        # with open('tca_imu_2.txt', "w") as f:
            # cm.print_matrix(stream=f)

        # 重定向 stat 输出到文件
        # with open('tca_imu_2.txt', "a") as f:  # 使用 'a' 模式以追加的形式写入
            # cm.stat(summary=True, stream=f)
        cm.print_matrix()
        cm.stat(summary=True)
        sys.stdout = sys.__stdout__           # Reset redirect.
        with open(f'{self.task}.txt', "w") as f:
            f.write(captured_output.getvalue())
            f.write(f"fp_label_dict: {self.fp_label_dict}\n")
            f.write(f"fn_label_dict: {self.fn_label_dict}\n")

            f.write("False Multi Detection:\n")
            for label in self.false_multi_dict:
                f.write(label)
                f.write(str(self.false_multi_dict[label]))
                f.write('\n')
        print('*' * 10 + "Testing Process" + '*' * 10)

    def configure_optimizers(self):
        # if self.ft_entire_network_beats:
        #     optimizer = optim.AdamW(
        #         [{"params": self.beats.parameters()}, {"params": self.fc.parameters()}],
        #         lr=self.lr, betas=(0.9, 0.98), weight_decay=0.01
        #     )  
        # else:
        params_list = []
        if 'imu' in self.task:
            params_list.append({'params': self.imu_feature_extractor.parameters()})
        # if 'two_channel' in self.task:
            # params_list.append({'params': self.beats.patch_embedding.parameters()})

        params_list.append({"params": self.beats.parameters()})
        params_list.append({'params': self.beats_feature_fc.parameters()})
        params_list.append({'params': self.fc.parameters()})

        optimizer = optim.AdamW(
            params_list,
            # self.fc.parameters(),
            lr=self.lr, betas=(0.9, 0.98), weight_decay=0.01
        )  

        return optimizer
    
class BEATsTransferLearningModel(pl.LightningModule):
    def __init__(
        self,
        num_target_classes: int = 50,
        milestones: int = 5,
        batch_size: int = 32,
        lr: float = 1e-3,
        lr_scheduler_gamma: float = 1e-1,
        num_workers: int = 6,
        model_path: str = "/model/BEATs_iter3_plus_AS2M.pt",
        ft_entire_network: bool = False, # Boolean on whether the classifier layer + BEATs should be fine-tuned
        **kwargs,
    ) -> None:
        """TransferLearningModel.
        Args:
            lr: Initial learning rate
        """
        super().__init__()
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.milestones = milestones
        self.num_target_classes = num_target_classes
        self.ft_entire_network = ft_entire_network

        # Initialise BEATs model
        self.checkpoint = torch.load(model_path)
        self.cfg = BEATsConfig(
            {
                **self.checkpoint["cfg"],
                "predictor_class": self.num_target_classes,
                "finetuned_model": False,
            }
        )

        self._build_model()

        self.train_acc = Accuracy(
            task="multiclass", num_classes=self.num_target_classes
        )
        self.valid_acc = Accuracy(
            task="multiclass", num_classes=self.num_target_classes
        )
        self.save_hyperparameters()

    def _build_model(self):
        # 1. Load the pre-trained network
        self.beats = BEATs(self.cfg)
        self.beats.load_state_dict(self.checkpoint["model"])

        # 2. Classifier
        self.fc = nn.Linear(self.cfg.encoder_embed_dim, self.cfg.predictor_class)

    def extract_features(self, x, padding_mask=None):
        if padding_mask != None:
            x, _ = self.beats.extract_features(x, padding_mask)
        else:
            x, _ = self.beats.extract_features(x)
        return x

    def forward(self, x, padding_mask=None):
        """Forward pass. Return x"""

        # Get the representation
        if padding_mask != None:
            x, _ = self.beats.extract_features(x, padding_mask)
        else:
            x, _ = self.beats.extract_features(x)

        # Get the logits
        x = self.fc(x)

        # Mean pool the second layer
        x = x.mean(dim=1)

        return x

    def loss(self, lprobs, labels):
        self.loss_func = nn.CrossEntropyLoss()
        return self.loss_func(lprobs, labels)

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, padding_mask, y_true = batch
        y_probs = self.forward(x, padding_mask)

        # 2. Compute loss
        train_loss = self.loss(y_probs, y_true)

        # 3. Compute accuracy:
        self.log("train_acc", self.train_acc(y_probs, y_true), prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, padding_mask, y_true = batch
        y_probs = self.forward(x)

        # 2. Compute loss
        self.log("val_loss", self.loss(y_probs, y_true), prog_bar=True)

        # 3. Compute accuracy:
        self.log("val_acc", self.valid_acc(y_probs, y_true), prog_bar=True)

    def configure_optimizers(self):
        if self.ft_entire_network:
            optimizer = optim.AdamW(
                [{"params": self.beats.parameters()}, {"params": self.fc.parameters()}],
                lr=self.lr, betas=(0.9, 0.98), weight_decay=0.01
            )  
        else:
            optimizer = optim.AdamW(
                self.fc.parameters(),
                lr=self.lr, betas=(0.9, 0.98), weight_decay=0.01
            )  


        return optimizer
