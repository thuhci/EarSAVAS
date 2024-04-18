#!/usr/bin/env python3

from pytorch_lightning import cli_lightning_logo
from pytorch_lightning.cli import LightningCLI

from fine_tune.transferLearning import BEATsEarVASTransferLearningModel
from datamodules.EarSAVASDataModule import EarSAVASDataModule
from callbacks.callbacks import MilestonesFinetuning


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(MilestonesFinetuning, "finetuning")
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("finetuning.milestones", "model.milestones")
        parser.set_defaults(
            {
                "trainer.max_epochs": 50,
                "trainer.enable_model_summary": True,
                "trainer.num_sanity_val_steps": 0,
            }
        )

def cli_main():
    MyLightningCLI(
        BEATsEarVASTransferLearningModel, EarSAVASDataModule, seed_everything_default=42
    )

if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()