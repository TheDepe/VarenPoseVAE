import os
import torch
import argparse
import trimesh

from torch.utils.data import DataLoader
from torch.optim import Adam

from src.models.varen_poser import VarenPoserTrainingExtension
from src.utils.logger import *
from src.datasets.varen_pose_dataset import VarenMoCapData

# --------------------------------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

# Model arguments
parser.add_argument('--varen_model_path', type=str, default="/home/dperrett/Documents/Data/VAREN/models/VAREN")

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='/ps/project/horses_in_motion/VARENset_subset') #'./data/shapenet.hdf5')
parser.add_argument('--train_batch_size', type=int, default=2048)
parser.add_argument('--val_batch_size', type=int, default=512)
parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint file")

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-5) #1e-5

# Training
parser.add_argument('--epochs', type=int, default=float('inf'))  # Specify the number of epochs

args = parser.parse_args()


# --------------------------------------------------------------------------------------------------


log_dir = get_new_log_dir()
logger = create_logger("VAE Trainer", log_dir)

logger.info('[Initialisation] Loading Datasets...')

train_data = VarenMoCapData("/home/dperrett/Documents/Data/VAREN/From Ci")

dataloader_train = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
device = 'cuda'

model = VarenPoserTrainingExtension(varen_path=args.varen_model_path).to(device)
params_to_optimize = [p for name, p in model.named_parameters() if 'body_model' not in name]
optimiser = Adam(params_to_optimize, lr=args.lr, weight_decay=0.00001)

# Load checkpoint if provided
start_epoch = 0
if args.checkpoint and os.path.exists(args.checkpoint):
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint, strict=False)

    logger.info(f"Resuming training from epoch {start_epoch}.")


logger.info(f'[Initialisation] Size of model: {sum(p.numel() for p in params_to_optimize)}')

kld_weight = 1.0 / len(dataloader_train)  # Normalization for KL divergence


epoch = 0
while epoch <= args.epochs:
    logger.info(f'Starting Epoch {epoch+1}/{args.epochs}')
    for iter, batch in enumerate(dataloader_train):
        batch = batch.to(device)
        batchsize = batch.shape[0]
        optimiser.zero_grad()
        recon = model(batch.float())
        loss_dict = model.compute_loss(batch, recon)
        loss = loss_dict['weighted_loss']['loss_total']
        loss.backward()

        optimiser.step()
        #scheduler.step()
        if iter % 1 == 0:
            logger.info(f'[Train] Iter {iter} | Loss {loss.item():0.4f} | ')
        
    if epoch % 1 == 0 or epoch == args.epochs:
        orig, recon = model.construct_meshes(batch, recon)
        orig = orig.vertices.detach().cpu().numpy()
        recon = recon.vertices.detach().cpu().numpy()
        
        idx = torch.randint(0, batchsize, (1,)).item()  # Extract the integer value
        trimesh.Trimesh(vertices=orig[idx], faces=model.body_model.faces).export(log_dir / f"input_epoch_{epoch}.ply")
        trimesh.Trimesh(vertices=recon[idx], faces=model.body_model.faces).export(log_dir / f"recon_epoch_{epoch}.ply")

        # Filter the state dict to exclude 'body_model' parameters
        state_dict = model.state_dict()

        # Create a new state dict excluding parameters from 'body_model'
        filtered_state_dict = {k: v for k, v in state_dict.items() if 'body_model' not in k}
        
        torch.save(filtered_state_dict, log_dir / f"VAREN_pose_VAE_epoch_{epoch}.pth")

    epoch += 1





