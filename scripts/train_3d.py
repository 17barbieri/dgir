#!/usr/bin/env python3
"""
3D Registration Training Script

This script trains a 3D diffusion-regularized registration network.
Based on your original train3d.py script.
"""

import argparse
import sys
import os
import random
import glob
import re
from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from diffusion_registration.training.config import Config
from diffusion_registration.core.models import setup_diffusion_model, setup_registration_net
from diffusion_registration.core import networks, wrappers, losses

from standard_utils import *

def dice_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Dice score between two binary masks.
    
    Args:
        y_true: Ground truth mask (0s and 1s)
        y_pred: Predicted mask (0s and 1s)
    
    Returns:
        Dice coefficient (float between 0 and 1)
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    
    intersection = np.logical_and(y_true, y_pred).sum()
    total = y_true.sum() + y_pred.sum()
    
    if total == 0:
        return 1.0  # both empty masks → perfect overlap
    return 2 * intersection / total

def compute_metric(fixed_seg, moving_seg):
    if isinstance(fixed_seg, nib.nifti1.Nifti1Image):
        fixed_seg = fixed_seg.get_fdata()
    if isinstance(moving_seg, nib.nifti1.Nifti1Image):
        moving_seg = moving_seg.get_fdata()
    dices_per_label = []
    for current_label in np.unique(fixed_seg):
        if current_label != 0:
            mask1 = fixed_seg == current_label
            mask2 = moving_seg == current_label
            dices_per_label.append(round(float(dice_score(mask1, mask2)), 3))

    mask1 = fixed_seg > 0
    mask2 = moving_seg > 0
    global_dice = dice_score(mask1, mask2)
    avg_dice = np.mean(dices_per_label)
    return global_dice, avg_dice, dices_per_label

class NiftiDataset(Dataset):
    """CSV with: Fixed path, Moving path, Fixed seg path, Moving seg path"""
    def __init__(self, csv_path, mode = 'train', target_shape = (224, 192, 160),
                 temp_save_path = './preproc_data'):
        self.dset = pd.read_csv(csv_path)
        self.mode = mode
        self.target_shape = target_shape
        self.temp_save_path = os.path.join(temp_save_path, mode)
        os.makedirs(self.temp_save_path, exist_ok = True)

    def __len__(self):
        return len(self.dset)

    def pad_or_crop_center(self, arr, target_shape, pad_value=0):
        """
        Pads or crops a 3D tensor (B, C, D, H, W) to match target_shape (D, H, W).
        """
        _, _, d, h, w = arr.shape
        td, th, tw = target_shape

        # Pad or crop depth
        pad_d = max(td - d, 0)
        pad_h = max(th - h, 0)
        pad_w = max(tw - w, 0)

        # Pad equally on both sides
        pad = (pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
            pad_d // 2, pad_d - pad_d // 2)
        arr = F.pad(arr, pad, mode='constant', value=pad_value)

        # Crop if needed (center crop)
        _, _, d, h, w = arr.shape
        start_d = (d - td) // 2
        start_h = (h - th) // 2
        start_w = (w - tw) // 2
        arr = arr[:, :, start_d:start_d+td, start_h:start_h+th, start_w:start_w+tw]

        return arr

    def __getitem__(self, idx):
        preproc_fixed_path = os.path.join(self.temp_save_path, f'{idx:03d}_fixed.pty')
        preproc_moving_path = os.path.join(self.temp_save_path, f'{idx:03d}_moving.pty')
        preproc_fixed_seg_path = os.path.join(self.temp_save_path, f'{idx:03d}_fixed_seg.pty')
        preproc_moving_seg_path = os.path.join(self.temp_save_path, f'{idx:03d}_moving_seg.pty')
        fixed_arr, fixed_seg_arr, moving_arr = None, None, None

        if os.path.isfile(preproc_fixed_path) and os.path.isfile(preproc_moving_path):
            fixed_crop_pad_tensor = torch.load(preproc_fixed_path)
            moving_crop_pad_tensor = torch.load(preproc_moving_path)
        if os.path.isfile(preproc_fixed_seg_path) and os.path.isfile(preproc_moving_seg_path):
            fixed_seg_crop_pad_tensor = torch.load(preproc_fixed_seg_path)
            moving_seg_crop_pad_tensor = torch.load(preproc_moving_seg_path)

        if self.mode == 'train':
            if fixed_arr is None:
                fixed_nib = nib_load(self.dset['Fixed path'].iloc[idx])
                moving_nib = nib_load(self.dset['Moving path'].iloc[idx])
                
                fixed_matched_nib, moving_matched_nib = match_nii_images(fixed_nib, moving_nib)

                fixed_arr = zscore_normalize(fixed_matched_nib).get_fdata()
                moving_arr = zscore_normalize(moving_matched_nib).get_fdata()

                fixed_tensor = torch.from_numpy(fixed_arr).unsqueeze(0).unsqueeze(0)
                moving_tensor = torch.from_numpy(moving_arr).unsqueeze(0).unsqueeze(0)

                # Pad or crop instead of interpolate
                fixed_crop_pad_tensor = self.pad_or_crop_center(fixed_tensor, self.target_shape, pad_value=0).squeeze(1).to(torch.float32)
                moving_crop_pad_tensor = self.pad_or_crop_center(moving_tensor, self.target_shape, pad_value=0).squeeze(1).to(torch.float32)

                torch.save(fixed_crop_pad_tensor, preproc_fixed_path)
                torch.save(moving_crop_pad_tensor, preproc_moving_path)

            return {'fixed_arr': fixed_crop_pad_tensor, 'moving_arr': moving_crop_pad_tensor}

        elif self.mode == 'val':
            if fixed_seg_arr is None:
                fixed_nib = nib_load(self.dset['Fixed path'].iloc[idx])
                moving_nib = nib_load(self.dset['Moving path'].iloc[idx])
                fixed_seg_nib = nib_load(self.dset['Fixed seg path'].iloc[idx])
                moving_seg_nib = nib_load(self.dset['Moving seg path'].iloc[idx])

                fixed_matched_nib, fixed_matched_seg_nib, moving_matched_nib, moving_matched_seg_nib = match_nii_images(fixed_nib, moving_nib, fixed_seg_nib, moving_seg_nib)

                fixed_arr = zscore_normalize(fixed_matched_nib).get_fdata()
                moving_arr = zscore_normalize(moving_matched_nib).get_fdata()

                fixed_tensor = torch.from_numpy(fixed_arr).unsqueeze(0).unsqueeze(0)
                fixed_seg_tensor = torch.from_numpy(fixed_matched_seg_nib.get_fdata()).unsqueeze(0).unsqueeze(0)
                moving_tensor = torch.from_numpy(moving_arr).unsqueeze(0).unsqueeze(0)
                moving_seg_tensor = torch.from_numpy(moving_matched_seg_nib.get_fdata()).unsqueeze(0).unsqueeze(0)

                # Pad or crop instead of interpolate
                fixed_crop_pad_tensor = self.pad_or_crop_center(fixed_tensor, self.target_shape, pad_value=0).squeeze(1).to(torch.float32)
                moving_crop_pad_tensor = self.pad_or_crop_center(moving_tensor, self.target_shape, pad_value=0).squeeze(1).to(torch.float32)
                fixed_seg_crop_pad_tensor = self.pad_or_crop_center(fixed_seg_tensor, self.target_shape, pad_value=0).squeeze(1).to(torch.float32)
                moving_seg_crop_pad_tensor = self.pad_or_crop_center(moving_seg_tensor, self.target_shape, pad_value=0).squeeze(1).to(torch.float32)

                torch.save(fixed_crop_pad_tensor, preproc_fixed_path)
                torch.save(moving_crop_pad_tensor, preproc_moving_path)
                torch.save(fixed_seg_crop_pad_tensor, preproc_fixed_seg_path)
                torch.save(moving_seg_crop_pad_tensor, preproc_moving_seg_path)

            return {
                'fixed_arr': fixed_crop_pad_tensor,
                'fixed_seg_arr': fixed_seg_crop_pad_tensor,
                'moving_arr': moving_crop_pad_tensor,
                'moving_seg_arr': moving_seg_crop_pad_tensor
            }

        else:
            raise NotImplementedError(f"Mode is not train/val: {self.mode}")

def create_3d_network(config, model, input_shape):
    """Create 3D registration network."""
    # Create base 3D network
    inner_net = wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))

    # Add multiscale levels
    for _ in range(3):
        inner_net = wrappers.TwoStepRegistration(
            wrappers.DownsampleRegistration(inner_net, dimension=3),
            wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))
        )

    # Create loss function
    if config.loss.type == "DINOFeatureLoss":
        loss_fn = losses.DINOFeatureLoss(model=model, sigma=config.loss.sigma)
    else:
        loss_fn = losses.LNCC(sigma=config.loss.sigma)

    # Create final network with diffusion regularization
    net = wrappers.DiffusionRegularizedNet(
        inner_net,
        loss_fn,
        lmbda=config.loss.lambda_regularization
    )

    net.assign_identity_map(input_shape)
    return net

def train_3d(net, train_loader, optimizer, device):
    """Training loop"""
    net.train()
    all_loss_total = 0.
    sim_loss_total = 0.
    reg_loss_total = 0.
    for batch in tqdm(train_loader):
        fixed_img, moving_img = batch['fixed_arr'].to(device), batch['moving_arr'].to(device)
        optimizer.zero_grad()
        loss_object = net(fixed_img, moving_img)
        loss_object.all_loss.backward()
        optimizer.step()
        all_loss_total +=loss_object.all_loss.item()
        sim_loss_total +=loss_object.similarity_loss.item()
        reg_loss_total +=loss_object.bending_energy_loss.item()

    return {
        'all_loss': all_loss_total/len(train_loader),
        'sim_loss': sim_loss_total/len(train_loader),
        'reg_loss': reg_loss_total/len(train_loader)
    }

def val_3d(net, val_loader, device):
    """Validation function"""
    net.eval()
    all_loss_total = 0.
    sim_loss_total = 0.
    reg_loss_total = 0.
    global_dice_total = 0.
    avg_dice_total = 0.
    dices_total = np.zeros(4)
    with torch.inference_mode(): # Faster than "torch.no_grad()"
        for batch in tqdm(val_loader):
            fixed_img, moving_img = batch['fixed_arr'].to(device), batch['moving_arr'].to(device)
            fixed_seg_img, moving_seg_img = batch['fixed_seg_arr'].to(device).squeeze(0).squeeze(0), batch['moving_seg_arr'].to(device)
            
            loss_object = net(fixed_img, moving_img)
            deformation_field = net.phi_AB_vectorfield.float()
            warped_moving_seg_img = net.as_function_seg(moving_seg_img.float())(deformation_field).squeeze(0).squeeze(0)

            global_dice, avg_dice, dices = compute_metric(fixed_seg_img.cpu().numpy(), warped_moving_seg_img.cpu().numpy())

            all_loss_total +=loss_object.all_loss.item()
            sim_loss_total +=loss_object.similarity_loss.item()
            reg_loss_total +=loss_object.bending_energy_loss.item()
            global_dice_total += global_dice
            avg_dice_total += avg_dice
            dices_total += dices
    return {
        'all_loss': all_loss_total/len(val_loader),
        'sim_loss': sim_loss_total/len(val_loader),
        'reg_loss': reg_loss_total/len(val_loader),
        'global_dice': global_dice_total/len(val_loader),
        'avg_dice': avg_dice_total/len(val_loader),
        'dices': dices_total/len(val_loader)
    }

def worker_init_fn(worker_id):
    """Worker_init_fn for dataloader initialization"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_checkpoint(config, net, optimizer, epoch, 
                    train_all_loss, train_sim_loss, train_reg_loss, 
                    val_all_loss, val_sim_loss, val_reg_loss, 
                    global_dice, avg_dice, dices, 
                    early_stopping_counter, best_val_loss):
    checkpoint_path = Path(config.output.checkpoint_dir) / f'net_3d_{epoch:04d}.pth'
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Collect checkpoint data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_metrics': {
            'all_loss': train_all_loss,
            'sim_loss': train_sim_loss,
            'reg_loss': train_reg_loss
        },
        'val_metrics': {
            'all_loss': val_all_loss,
            'sim_loss': val_sim_loss,
            'reg_loss': val_reg_loss,
            'global_dice': global_dice,
            'avg_dice': avg_dice,
            'dices': dices
        },
        'config': vars(config) if hasattr(config, '__dict__') else str(config),
        'seed': 42,
        'early_stopping_counter': early_stopping_counter,
        'best_val_loss': best_val_loss
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def main(config):
    """Main function."""

    # Set random seeds for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # For GPU determinism
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # (Optional but good practice)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Needed for CUDA >= 10.2 determinism

    device = torch.device(config.training.device)

    # Load data
    input_shape = [1, 1, 224, 192, 160]  # Standard 3D shape
    # input_shape = [1, 1, 256, 256, 80]  # Standard 3D shape
    train_ds = NiftiDataset(config.data.train_dset, mode = 'train', target_shape = input_shape[2:])
    val_ds = NiftiDataset(config.data.val_dset, mode = 'val', target_shape = input_shape[2:])
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, 
                              num_workers=config.training.num_workers, worker_init_fn=worker_init_fn, 
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=config.training.num_workers, worker_init_fn=worker_init_fn,
                            generator=torch.Generator().manual_seed(SEED))

    # Setup diffusion model
    # print("Setting up diffusion model...")
    # model, diffusion = setup_diffusion_model(config)
    # model = model.eval().to(device)

    print("Loading DINO-v2 model...")
    local_repo = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
    model = torch.hub.load(str(local_repo), 'dinov2_vitg14', source='local')
    model.eval().to(device)

    # Create network
    print("Creating 3D registration network...")
    # net = create_3d_network(config, model, diffusion, input_shape)
    net = create_3d_network(config, model, input_shape)
    net = net.to(device)
    net.train()
    print(f"Network parameters: {sum(p.numel() for p in net.parameters()):,}")

    # Setup optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=config.training.learning_rate)

    # Training loop
    train_all_loss = []
    train_sim_loss = []
    train_reg_loss = []
    val_all_loss = []
    val_sim_loss = []
    val_reg_loss = []
    global_dice = []
    avg_dice = []
    dices = []

    # Search last saved epoch
    # ckpt_list = glob.glob(os.path.join(config.output.checkpoint_dir, '*.pth'))
    ckpt_dir = Path(config.output.checkpoint_dir)
    ckpt_list = [str(f) for f in ckpt_dir.glob("*.pth") if re.search(r'\d+\.pth$', f.name)]
    if ckpt_list:
        ckpt_path = sorted(ckpt_list)[-1]
        resume_epoch = int(ckpt_path.split('_')[-1].split('.')[0])+1
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        train_all_loss = ckpt['train_metrics']['all_loss']
        train_sim_loss = ckpt['train_metrics']['sim_loss']
        train_reg_loss = ckpt['train_metrics']['reg_loss']
        val_all_loss = ckpt['val_metrics']['all_loss']
        val_sim_loss = ckpt['val_metrics']['sim_loss']
        val_reg_loss = ckpt['val_metrics']['reg_loss']
        global_dice = ckpt['val_metrics']['global_dice']
        avg_dice = ckpt['val_metrics']['avg_dice']
        dices = ckpt['val_metrics']['dices']

        early_stopping_counter = ckpt['early_stopping_counter']
        best_val_loss = ckpt['best_val_loss']
        print(f'Loaded checkpoint: {ckpt_path}')
        print(f"Resuming training at epoch: {resume_epoch}/{config.training.epochs}...")
        print(f'Best validation loss {best_val_loss:.5f}')
        print(f"Early stopping counter: {early_stopping_counter}/{config.training.early_stopping_patience}")
    else:
        print(f"Starting training for {config.training.epochs} epochs...")
        resume_epoch=1
        best_val_loss = float('inf')
        early_stopping_counter = 0.
    print(f"Batch size: {config.training.batch_size}")

    # Train model
    try:
        for epoch in range(resume_epoch, config.training.epochs+1):
            if early_stopping_counter>=config.training.early_stopping_patience:
                epoch-=1
                break
            train_metrics = train_3d(net, train_loader, optimizer, device)
            val_metrics = val_3d(net, val_loader, device)
            
            # Record losses
            train_all_loss.append(train_metrics['all_loss'])
            train_sim_loss.append(train_metrics['sim_loss'])
            train_reg_loss.append(train_metrics['reg_loss'])

            val_all_loss.append(val_metrics['all_loss'])
            val_sim_loss.append(val_metrics['sim_loss'])
            val_reg_loss.append(val_metrics['reg_loss'])
            global_dice.append(val_metrics['global_dice'])
            avg_dice.append(val_metrics['avg_dice'])
            dices.append(val_metrics['dices'])

            # Periodic logging
            print(f"[{epoch}] Train total: {train_all_loss[-1]:.4f}, Sim: {train_sim_loss[-1]:.4f}, Reg: {train_reg_loss[-1]:.4f}")
            print(f"[{epoch}] Val total: {val_all_loss[-1]:.4f}, Sim: {val_sim_loss[-1]:.4f}, Reg: {val_reg_loss[-1]:.4f}")
            print(f"[{epoch}] Global dice: {float(global_dice[-1]):.4f}, Avg dice: {float(avg_dice[-1]):.4f}")

            if epoch>30:
                if best_val_loss > val_all_loss[-1]:
                    best_val_loss = val_all_loss[-1]
                    early_stopping_counter = 0
                    print(f'New best validation loss: {val_all_loss[-1]}')
                    save_checkpoint(config, net, optimizer, epoch,
                                train_all_loss, train_sim_loss, train_reg_loss,
                                val_all_loss, val_sim_loss, val_reg_loss,
                                global_dice, avg_dice, dices, early_stopping_counter, best_val_loss)
                else:
                    early_stopping_counter+=1
                    print(f'Early stopping {early_stopping_counter}/{config.training.early_stopping_patience}')
                    if early_stopping_counter>=config.training.early_stopping_patience:
                        save_checkpoint(config, net, optimizer, epoch,
                                train_all_loss, train_sim_loss, train_reg_loss,
                                val_all_loss, val_sim_loss, val_reg_loss,
                                global_dice, avg_dice, dices, early_stopping_counter, best_val_loss)
                        print(f"Early stopping triggered at epoch {epoch}")
                        break

            # Save checkpoint
            if (epoch % config.training.save_every == 0) or (epoch == config.training.epochs):
                save_checkpoint(config, net, optimizer, epoch,
                                train_all_loss, train_sim_loss, train_reg_loss,
                                val_all_loss, val_sim_loss, val_reg_loss,
                                global_dice, avg_dice, dices, early_stopping_counter, best_val_loss)

        # Save final model
        final_path = Path(config.output.checkpoint_dir) / 'net_3d_final.pth'
        torch.save(net.state_dict(), final_path)
        print(f"Final model saved: {final_path}")
        print("Training completed successfully!")

        # Plot losses if matplotlib available
        try:
            _, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].plot(np.arange(1, epoch+1), train_all_loss, label='Train')
            axes[0].plot(np.arange(1, epoch+1), val_all_loss, label = 'Val')
            axes[0].set_title('Total Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].grid(True)
            axes[0].legend()

            axes[1].plot(np.arange(1, epoch+1), train_sim_loss, label='Train')
            axes[1].plot(np.arange(1, epoch+1), val_sim_loss, label = 'Val')
            axes[1].set_title('Similarity Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].grid(True)
            axes[1].legend()

            axes[2].plot(np.arange(1, epoch+1), train_reg_loss, label='Train')
            axes[2].plot(np.arange(1, epoch+1), val_reg_loss, label = 'Val')
            axes[2].set_title('Regularization Loss')
            axes[2].set_xlabel('Epoch')
            axes[2].grid(True)
            axes[2].legend()

            plt.tight_layout()
            if config.training.random_weights:
                fig_path = Path(config.output.results_dir) / 'random_weights_training_losses_3d.png'
            else:
                fig_path = Path(config.output.results_dir) / 'training_losses_3d.png'
            plt.savefig(fig_path)
            plt.close()
            print(f"Loss plot saved to: {fig_path}")

            plt.title('Average dice')
            plt.plot(np.arange(1, epoch+1), avg_dice)
            plt.xlabel('Epoch')
            plt.grid(True)
            if config.training.random_weights:
                fig_path = Path(config.output.results_dir) / 'random_weights_validation_dice_3d.png'
            else:
                fig_path = Path(config.output.results_dir) / 'training_validation_dice_3d.png'
            plt.savefig(fig_path)
            print(f"Loss plot saved to: {fig_path}")

        except ImportError:
            print("matplotlib not available, skipping loss plot")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Training failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train 3D diffusion registration network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )
    parser.add_argument('--config', type=str, default='config/config_3d.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--random-weights', action= 'store_true',
                        help = 'Runs the network without initializing the diffusion model weights')
    parser.add_argument('--data-root', type=str, default = '/mnt/c/Users/IMAG2/Documents/MATTEO/Data',
                        help='Override data root directory')
    parser.add_argument('--train-dset', type=str, default = './splits/dset_train.csv',
                        help='Override data root directory')
    parser.add_argument('--val-dset', type=str, default = './splits/dset_val.csv',
                        help='Override data root directory')
    parser.add_argument('--test-dset', type=str, default = './splits/dset_test.csv',
                        help='Override data root directory')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers')
    parser.add_argument('--epochs', type=int,default=1000,
                        help='Override number of epochs')
    parser.add_argument('--early-stopping-patience', type=int,default=30,
                        help='Early stopping patience')
    parser.add_argument('--save-every', type=int,default=5,
                        help='Override saving checkpoint frequency')
    parser.add_argument('--lambda-reg', type=float, default=0.5,
                        help='Regularization weight')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Override learning rate')
    parser.add_argument('--loss-type', type=str, default='NewLNCC3D',
                        choices=['NewLNCC3D', 'LNCC'], help='Loss function type')
    args = parser.parse_args()
    
    # Load configuration
    configuration = Config(args.config)
    os.makedirs(configuration.output.results_dir, exist_ok = True)

    # Override config with command line arguments
    configuration.training.epochs = args.epochs
    configuration.training.save_every = args.save_every
    configuration.loss.lambda_regularization = args.lambda_reg
    configuration.loss.type = args.loss_type

    if args.data_root:
        configuration.data.data_root = args.data_root
    if args.train_dset:
        configuration.data.train_dset= args.train_dset
    if args.val_dset:
        configuration.data.val_dset = args.val_dset
    if args.test_dset:
        configuration.data.test_dset = args.test_dset
    configuration.training.num_workers = args.num_workers
    if args.learning_rate:
        configuration.training.learning_rate = args.learning_rate
    configuration.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    configuration.training.random_weights = args.random_weights
    configuration.training.early_stopping_patience = args.early_stopping_patience
    if configuration.training.random_weights:
        configuration.output.checkpoint_dir = 'random_diffusion_weights_' + configuration.output.checkpoint_dir
    

    # Create csv files
    b0_files = glob.glob(os.path.join(args.data_root, 'b0', '*b0.nii.gz'))
    b0_seg_files = glob.glob(os.path.join(args.data_root, 'b0_seg_corrected_labels/*b0_seg.nii.gz'))
    all_patient_codes = [os.path.basename(p).split('_b0')[0] for p in b0_files]
    val_patient_codes = [os.path.basename(p).split('_b0')[0] for p in b0_seg_files]

    train_dset = {'Patient code': [], 'Fixed path': [], 'Moving path': []}
    val_dset = {'Patient code': [],
                'Fixed path': [], 'Fixed seg path': [],
                'Moving path': [], 'Moving seg path': []}

    for patient_code in all_patient_codes:
        if patient_code in val_patient_codes:
            val_dset['Patient code'].append(patient_code)
            val_dset['Fixed path'].append(glob.glob(os.path.join(args.data_root, 'b0', f'{patient_code}_b0.nii.gz'))[0])
            val_dset['Fixed seg path'].append(glob.glob(os.path.join(args.data_root, 'b0_seg_corrected_labels', f'{patient_code}_b0_seg.nii.gz'))[0])
            val_dset['Moving path'].append(glob.glob(os.path.join(args.data_root, 'T2', f'{patient_code}_coroT2cube.nii.gz'))[0])
            val_dset['Moving seg path'].append(glob.glob(os.path.join(args.data_root, 'T2_seg', f'{patient_code}_segmentation.nii.gz'))[0])
        else:
            train_dset['Patient code'].append(patient_code)
            train_dset['Fixed path'].append(glob.glob(os.path.join(args.data_root, 'b0', f'{patient_code}_b0.nii.gz'))[0])
            train_dset['Moving path'].append(glob.glob(os.path.join(args.data_root, 'T2', f'{patient_code}_coroT2cube.nii.gz'))[0])
    os.makedirs(os.path.dirname(args.train_dset), exist_ok = True)
    pd.DataFrame(train_dset).to_csv(args.train_dset, index = False)
    pd.DataFrame(val_dset).to_csv(args.val_dset, index = False)

    print("Starting 3D registration training with:")
    print(f"  Config: {args.config}")
    print(f"  Device: {configuration.training.device}")
    print(f"  Batch size: {configuration.training.batch_size}")
    print(f"  Learning rate: {configuration.training.learning_rate}")
    print(f"  Epochs: {configuration.training.epochs}")
    print(f"  Loss type: {configuration.loss.type}")
    print(f"  Lambda regularization: {configuration.loss.lambda_regularization}")

    main(configuration)
