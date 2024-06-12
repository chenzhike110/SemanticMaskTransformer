import os
import random
import argparse
from omegaconf import OmegaConf
from tools.utils import instantiate_from_config, makepath

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar, EarlyStopping

from transformers import logging
logging.set_verbosity_error()

def parse_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group("Training options")
    group.add_argument(
        "--cfg",
        type=str,
        required=False,
        help="config file",
    )
    group.add_argument(
        "--resume",
        type=str,
        default=None,
        required=False,
        help="resume directory",
    )
    params = parser.parse_args()

    if params.resume:
        resume = params.resume
        if os.path.exists(resume):
            file_list = sorted(os.listdir(resume), reverse=True)
            for item in file_list:
                if item.endswith(".yaml"):
                    cfg = OmegaConf.load(os.path.join(resume, item))
                    break
            cfg.train.seed = resume.split('/')[-1]
            checkpoints = sorted(os.listdir(os.path.join(
                resume, "checkpoints")),
                                 key=lambda x: int(x[6:-5]),
                                 reverse=True)
            for checkpoint in checkpoints:
                if "epoch=" in checkpoint:
                    cfg.train.pretrained = os.path.join(
                        resume, "checkpoints", checkpoint)
                    break
            if os.path.exists(os.path.join(resume, "wandb")):
                wandb_list = sorted(os.listdir(os.path.join(resume, "wandb")),
                                    reverse=True)
                for item in wandb_list:
                    if "run-" in item:
                        cfg.logger.resume_id = item.split("-")[-1]

        else:
            raise ValueError("Resume path is not right.")

    else:
        cfg = OmegaConf.load(params.cfg)

    return cfg

def main():
    cfg = parse_args()
    
    model = instantiate_from_config(cfg)

    # set seed
    if not hasattr(cfg.train, 'seed'):
        cfg.train.seed = random.randint(0, 25536)
    pl.seed_everything(cfg.train.seed)

    cfg.folder_exp = os.path.join("output", cfg.logger.project, str(cfg.train.seed))

    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        offline=cfg.logger.offline,
        id=cfg.logger.resume_id,
        save_dir=cfg.folder_exp,
        version="",
        name=cfg.name,
        anonymous=False,
        log_model=False,
    )

    callbacks = [
        RichProgressBar(),
        LearningRateMonitor(),
        ModelCheckpoint(
            dirpath=makepath(os.path.join(cfg.train.saved, "checkpoints"), isfile=True),
            save_top_k=1,
            monitor='val_acc',
            mode='max',
            save_last=True,
            every_n_epochs=cfg.train.val_frequency,
        ),
    ]