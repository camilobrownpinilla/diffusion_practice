from ddpm_config import BaseConfig as BConfig
from ddpm_config import TrainingConfig as TConfig
import dependencies
import helpers
import os
import torch
from unet import UNet
import ddpm_config as ddpm
from ddpm_train import ModelConfig
import ddpm_train
import ddpm_practice_implementation

model = UNet(
    input_channels          = TConfig.IMG_SHAPE[0],
    output_channels         = TConfig.IMG_SHAPE[0],
    base_channels           = ModelConfig.BASE_CH,
    base_channels_multiples = ModelConfig.BASE_CH_MULT,
    apply_attention         = ModelConfig.APPLY_ATTENTION,
    dropout_rate            = ModelConfig.DROPOUT_RATE,
    time_multiple           = ModelConfig.TIME_EMB_MULT,
)


model.load_state_dict(torch.load(os.path.join(BConfig.checkpoint_dir, "ckpt.tar"), map_location='cpu')['model'])

model.to(BConfig.DEVICE)

sd = SimpleDiffusion(
    num_diffusion_timesteps = TrainingConfig.TIMESTEPS,
    img_shape               = TrainingConfig.IMG_SHAPE,
    device                  = BaseConfig.DEVICE,
)

log_dir = "inference_results"
os.makedirs(log_dir, exist_ok=True)

generate_video = False # Set it to True for generating video of the entire reverse diffusion proces or False to for saving only the final generated image.
 
ext = ".mp4" if generate_video else ".png"
filename = f"{dependencies.datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}"
 
save_path = os.path.join(BConfig.log_dir, filename)
 
ddpm_train.reverse_diffusion(
    ddpm_train.model,
    ddpm_practice_implementation.DenoiseDiffusion(ddpm_train.model, TConfig.TIMESTEPS, BConfig.DEVICE),
    num_images=256,
    generate_video=generate_video,
    save_path=save_path,
    timesteps=2,
    img_shape=TConfig.IMG_SHAPE,
    device=BConfig.DEVICE,
    nrow=32,
)
print(save_path)