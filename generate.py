from ddpm_config import BaseConfig as BConfig
from ddpm_config import TrainingConfig as TConfig
import os
import ddpm_config as ddpm
import ddpm_train
import ddpm_practice_implementation

generate_video = True # Set it to True for generating video of the entire reverse diffusion proces or False to for saving only the final generated image.
 
ext = ".mp4" if generate_video else ".png"
filename = f"yup"
 
save_path = os.path.join(BConfig.log_dir, filename)
 
ddpm_train.reverse_diffusion(
    ddpm_train.model,
    ddpm_practice_implementation.DenoiseDiffusion(ddpm_train.model, TConfig.TIMESTEPS, BConfig.DEVICE),
    num_images=256,
    generate_video=generate_video,
    save_path=save_path,
    timesteps=1000,
    img_shape=TConfig.IMG_SHAPE,
    device=BConfig.DEVICE,
    nrow=32,
)
print(save_path)