from dataclasses import dataclass
import torch
import helpers
import os

@dataclass
class BaseConfig:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DATASET = "CIFAR-10"

    working_dir = os.getcwd()
    root_log_dir = os.path.join(working_dir, "Logs_Checkpoints", "Inference")
    root_checkpoint_dir = os.path.join("Logs_Checkpoints", "checkpoints")

    recent_version = helpers.get_most_recent_version(root_checkpoint_dir)
    log_dir = os.path.join(root_log_dir, recent_version)
    checkpoint_dir = os.path.join(root_checkpoint_dir, recent_version)

class TrainingConfig:
    TIMESTEPS = 1000
    IMG_SHAPE = (3, 32, 32)
    NUM_EPOCHS = 800
    BATCH_SIZE = 32
    LR = 2e-4
    NUM_WORKERS = 2
