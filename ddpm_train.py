from ddpm_config import TrainingConfig as TConfig
from ddpm_config import BaseConfig as BConfig
import ddpm_data
import ddpm_practice_implementation
import helpers
from helpers import get
from unet import UNet
from dependencies import *
# import ddpm_data
# import ddpm_practice_implementation
# import unet
# import torch
# import helpers
# from torchmetrics import MeanMetric
# from tqdm import tqdm
# from torch.cuda import amp

def train_one_epoch(model, loader, sd, optimizer, scaler, loss_fn, epoch=800, 
                   base_config=BConfig(), training_config=TConfig()):
     
    loss_record = MeanMetric()
    model.train()
 
    with tqdm(total=len(loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{training_config.NUM_EPOCHS}")
          
        for x0s, _ in loader: # line 1, 2
            tq.update(1)
             
            ts = torch.randint(low=1, high=training_config.TIMESTEPS, size=(x0s.shape[0],), device=BConfig.DEVICE) # line 3
            xts, gt_noise = sd.q_sample(x0s, ts) # line 4
 
            with amp.autocast():
                pred_noise = model(xts, ts)
                loss = loss_fn(gt_noise, pred_noise) # line 5
 
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
 
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
 
            scaler.step(optimizer)
            scaler.update()
 
            loss_value = loss.detach().item()
            loss_record.update(loss_value)
 
            tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")
 
        mean_loss = loss_record.compute().item()
     
        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")
     
    return mean_loss

@torch.no_grad()
def reverse_diffusion(model, sd, timesteps=1000, img_shape=(3, 64, 64), 
                      num_images=5, nrow=8, device="cpu", **kwargs):
    
    x = torch.randn((num_images, *img_shape), device=device)
    model.eval()
    
    save_path = BConfig.working_dir + "/sample.png"
    if kwargs.get("generate_video", False):
        outs = []
 
    for time_step in tqdm(iterable=reversed(range(1, timesteps)), 
                          total=timesteps-1, dynamic_ncols=False, 
                          desc="Sampling :: ", position=0):
 
        ts = torch.ones(num_images, dtype=torch.long, device=device) * time_step
        z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)
 
        predicted_noise = model(x, ts)
 
        beta_t                            = get(sd.beta, ts)
        one_by_sqrt_alpha_t               = get(sd.one_by_sqrt_alpha, ts)
        sqrt_one_minus_alpha_cumulative_t = get(sd.sqrt_one_minus_alpha_cumulative, ts) 
 
        x = (
            one_by_sqrt_alpha_t
            * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
            + torch.sqrt(beta_t) * z
        )
 
        if kwargs.get("generate_video", False):
            x_inv = ddpm_data.inverse_transform(x).type(torch.uint8)
            grid = make_grid(x_inv, nrow=nrow, pad_value=255.0).to("cpu")
            ndarr = torch.permute(grid, (1, 2, 0)).numpy()[:, :, ::-1]
            outs.append(ndarr)
 
    if kwargs.get("generate_video", False): # Generate and save video of the entire reverse process. 
        helpers.frames2vid(outs, kwargs['save_path'])
        helpers.display(Image.fromarray(outs[-1][:, :, ::-1])) # Display the image at the final timestep of the reverse process.
        return None
 
    else: # Display and save the image at the final timestep of the reverse process. 
        x = ddpm_data.inverse_transform(x).type(torch.uint8)
        grid = make_grid(x, nrow=nrow, pad_value=255.0).to("cpu")
        pil_image = TF.functional.to_pil_image(grid)
        pil_image.save(kwargs['save_path'], format=save_path[-3:].upper())
        display(pil_image)
        return None

class ModelConfig:
    BASE_CH = 64
    BASE_CH_MULT = (1, 2, 4, 4)
    APPLY_ATTENTION = (False, True, True, False)
    DROPOUT_RATE = 0.1
    TIME_EMB_MULT = 4

model = UNet(
    input_channels = TConfig.IMG_SHAPE[0],
    output_channels = TConfig.IMG_SHAPE[0],
    base_channels = ModelConfig.BASE_CH,
    base_channels_multiples = ModelConfig.BASE_CH_MULT,
    apply_attention = ModelConfig.APPLY_ATTENTION,
    dropout_rate = ModelConfig.DROPOUT_RATE,
    time_multiple = ModelConfig.TIME_EMB_MULT
)
model.to(BConfig.DEVICE)

if __name__ == '__main__':
    optimizer = torch.optim.AdamW(model.parameters(), lr=TConfig.LR)

    dataloader = ddpm_data.get_dataloader(
        dataset_name = BConfig.DATASET,
        batch_size = TConfig.BATCH_SIZE,
        device = BConfig.DEVICE,
        pin_memory = True,
        num_workers = TConfig.NUM_WORKERS
    )

    loss_fn = nn.MSELoss()

    diffusion = ddpm_practice_implementation.DenoiseDiffusion(model, TConfig.TIMESTEPS, BConfig.DEVICE)

    scaler = amp.GradScaler()

    total_epochs = TConfig.NUM_EPOCHS + 1
    log_dir, checkpoint_dir = helpers.setup_log_directory(config=BConfig())
    generate_video = False
    ext = ".mp4" if generate_video else ".png"


    """
    TRAINING LOOP
    """
    for epoch in range(1, total_epochs):
        torch.cuda.empty_cache()
        gc.collect()

        train_one_epoch(model, dataloader, diffusion, optimizer, scaler, loss_fn, epoch=epoch)

        if epoch % 20 == 0:
            save_path = os.path.join(log_dir, f"{epoch}{ext}")

            reverse_diffusion(model, diffusion, timesteps=TConfig.TIMESTEPS,
                            num_images=32, generate_video=generate_video,
                            save_path=save_path, img_shape=TConfig.IMG_SHAPE,
                            device=BConfig.DEVICE, nrow=4)
            
            checkpoint_dict = {
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "mode": model.state_dict()
            }
            torch.save(checkpoint_dict, os.path.join(checkpoint_dir, "ckpt.pt"))
            del checkpoint_dict