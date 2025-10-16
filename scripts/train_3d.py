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
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from nibabel.processing import resample_from_to
from tqdm import tqdm
import torch
import itk
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from diffusion_registration.training.config import Config
from diffusion_registration.core.models import setup_diffusion_model, setup_registration_net
from diffusion_registration.core import networks, wrappers, losses

def nib_load(path, raw = True, dtype = np.float32, orientation=('R', 'A', 'S')):
    """ Loads Nifti volumes and reorients them in ()'R', 'A', 'S')"""
    volume = nib.load(path)

    wanted_orientation = nib.orientations.axcodes2ornt(orientation)
    current_orientation = nib.orientations.io_orientation(volume.affine)
    if np.any(current_orientation != wanted_orientation):
        transformation = nib.orientations.ornt_transform(current_orientation, wanted_orientation)
        volume = volume.as_reoriented(transformation)
    volume = nib.Nifti1Image(volume.get_fdata().astype(dtype), affine = volume.affine, header = volume.header)

    if raw:
        return volume
    return volume, volume.get_fdata()

def zscore_normalize(volume_nib: nib.nifti1.Nifti1Image, eps=1e-8,
                     target_shape = (256, 256, 75)) -> np.ndarray:
    """Apply zscore normalization and reshape data"""
    # compute mean / std only over non-zero voxels (common for brain MRI)
    volume = volume_nib.get_fdata().copy()
    mask = volume != 0
    if mask.sum() == 0:
        v = volume
        mean = v.mean()
        std = v.std()
    else:
        v = volume[mask]
        mean = v.mean()
        std = v.std()
    if std < eps:
        std = eps
    out = (volume - mean) / std
    # clip extreme values for stability
    out = np.clip(out, -5.0, 5.0)

    # Resize if target_shape is given
    if target_shape is not None:
        zoom_factors = [t/s for t,s in zip(target_shape, out.shape)]
        out = zoom(out, zoom_factors, order=1)
    return out.astype(np.float32)

def dipy_resample(moving_img, fixed_img, moving_seg = None):
    """
    Resample a moving image and its segmentation to match a fixed image
    in world space, including orientation, origin, and spacing.

    Parameters
    ----------
    moving_img : nib.Nifti1Image
        Moving image to resample.
    moving_seg : nib.Nifti1Image
        Moving segmentation (integer labels).
    fixed_img : nib.Nifti1Image
        Fixed image that defines the target space.
    interp_type : str
        Interpolation type for image: 'linear' or 'nearest' (segmentation uses nearest).

    Returns
    -------
    resampled_data : np.ndarray
        Resampled moving image data (float).
    resampled_affine : np.ndarray
        Affine of resampled moving image.
    resampled_seg : np.ndarray
        Resampled segmentation data (int).
    """

    # 1. Resample the moving image to fixed image using world-space alignment
    # order = 1 --> Linear for image
    resampled_moving_img = resample_from_to(moving_img, fixed_img, order=1)

    # 2. Resample the segmentation (nearest neighbor)
    if moving_seg is not None:
        # order = 0 --> Nearest-neighbor for segmentation
        resampled_moving_seg = resample_from_to(moving_seg, fixed_img, order=0)

        return resampled_moving_img, resampled_moving_seg
    return resampled_moving_img

class NiftiDataset(Dataset):
    """CSV with: Fixed path, Moving path, Fixed seg path, Moving seg path"""
    def __init__(self, csv_path, mode = 'train'):
        self.dset = pd.read_csv(csv_path)
        self.mode = mode

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        target_shape = (224, 192, 160)

        fixed_nib = nib_load(self.dset['Fixed path'].iloc[idx])
        moving_nib = nib_load(self.dset['Moving path'].iloc[idx])

        if self.mode == 'train':
            moving_nib, _ = dipy_resample(moving_nib, moving_nib, fixed_nib)
            fixed_arr = zscore_normalize(fixed_nib)
            moving_arr = zscore_normalize(moving_nib)
            fixed_arr = torch.from_numpy(fixed_arr).unsqueeze(0).unsqueeze(0)
            moving_arr = torch.from_numpy(moving_arr).unsqueeze(0).unsqueeze(0)
            fixed_arr = F.interpolate(fixed_arr, size=target_shape, mode='trilinear', align_corners=False).squeeze(1)
            moving_arr = F.interpolate(moving_arr, size=target_shape, mode='trilinear', align_corners=False).squeeze(1)
            return {'fixed_arr': fixed_arr, 'moving_arr': moving_arr}
        elif self.mode =='val':
            fixed_seg_nib = nib_load(self.dset['Fixed seg path'].iloc[idx])
            moving_seg_nib = nib_load(self.dset['Moving seg path'].iloc[idx])
            moving_nib, moving_seg_nib = dipy_resample(moving_nib, moving_seg_nib, fixed_nib)
            fixed_arr = zscore_normalize(fixed_nib)
            moving_arr = zscore_normalize(moving_nib)
            fixed_arr = torch.from_numpy(fixed_arr).unsqueeze(0).unsqueeze(0)
            fixed_seg_arr = torch.from_numpy(fixed_seg_nib.get_fdata()).unsqueeze(0).unsqueeze(0)
            moving_arr = torch.from_numpy(moving_arr).unsqueeze(0).unsqueeze(0)
            moving_seg_arr = torch.from_numpy(moving_seg_nib.get_fdata()).unsqueeze(0).unsqueeze(0)
            fixed_arr = F.interpolate(fixed_arr, size=target_shape, mode='trilinear', align_corners=False).squeeze(1)
            moving_arr = F.interpolate(moving_arr, size=target_shape, mode='trilinear', align_corners=False).squeeze(1)
            fixed_seg_arr = F.interpolate(fixed_seg_arr, size=target_shape, mode='trilinear', align_corners=False).squeeze(1)
            moving_seg_arr = F.interpolate(moving_seg_arr, size=target_shape, mode='trilinear', align_corners=False).squeeze(1)
            return {'fixed_arr': fixed_arr, 'fixed_seg': fixed_seg_arr,
                    'moving_arr': moving_arr,
                    'moving_seg_arr': moving_seg_arr}

def load_3d_oasis_data(max_subjects=300, data_pattern=None):
    """
    Load 3D OASIS dataset.

    Args:
        max_subjects: Maximum number of subjects to load
        data_pattern: Pattern for data files (optional)

    Returns:
        Tuple of (dataset_A, dataset_B) as torch tensors
    """
    print(f"Loading 3D OASIS data (max {max_subjects} subjects)...")

    dataset_A = []
    dataset_B = []
    count = 0

    base_path = '/path-to-OASIS'
    if data_pattern:
        base_path = str(Path(data_pattern).parent.parent)

    for i in range(1, 458):
        if count >= max_subjects:
            break

        norm_path = f'{base_path}/OASIS_OAS1_{i:04}_MR1/aligned_norm.nii.gz'
        orig_path = f'{base_path}/OASIS_OAS1_{i:04}_MR1/aligned_orig.nii.gz'

        if not os.path.exists(norm_path) or not os.path.exists(orig_path):
            continue

        try:
            img_A = itk.imread(norm_path)
            img_B = itk.imread(orig_path)

            dataset_A.append(torch.from_numpy(np.asarray(img_A))[None])
            dataset_B.append(torch.from_numpy(np.asarray(img_B))[None])
            count += 1

            if count % 50 == 0:
                print(f"Loaded {count} subjects...")

        except Exception as e:
            print(f"Warning: Failed to load subject {i}: {e}")
            continue

    print(f"Successfully loaded {count} subjects")

    # Stack into tensors
    dataset_A = torch.stack(dataset_A)
    dataset_B = torch.stack(dataset_B)

    return dataset_A, dataset_B


def create_3d_network(config, model, diffusion, input_shape):
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
    if config.loss.type == "NewLNCC3D":
        loss_fn = losses.NewLNCC3D(
            diffusion=diffusion,
            model=model,
            sigma=config.loss.sigma
        )
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


def train_3d(config):
    """Main 3D training function."""
    device = torch.device(config.training.device)

    # Load data
    train_ds = NiftiDataset(config.data.train_dset, mode = 'train')
    val_ds = NiftiDataset(config.data.val_dset, mode = 'val')
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.training.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.training.num_workers)

    # Setup diffusion model
    print("Setting up diffusion model...")
    model, diffusion = setup_diffusion_model(config)
    model = model.eval().to(device)

    # Create network
    print("Creating 3D registration network...")
    input_shape = [1, 1, 224, 192, 160]  # Standard 3D shape
    net = create_3d_network(config, model, diffusion, input_shape)
    net = net.to(device)
    net.train()

    print(f"Network parameters: {sum(p.numel() for p in net.parameters()):,}")

    # Setup optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=config.training.learning_rate)

    # Training loop
    all_loss = []
    sim_loss = []
    reg_loss = []

    print(f"Starting training for {config.training.epochs} epochs...")
    print(f"Batch size: {config.training.batch_size}")

    for epoch in range(config.training.epochs):
        # Sample random batch
        all_loss_total = 0.
        sim_loss_total = 0.
        reg_loss_total = 0.
        for batch in tqdm(train_loader, desc=f'Running epoch {epoch}'):
            fixed_img, moving_img = batch['fixed_arr'].to(device), batch['moving_arr'].to(device)
            optimizer.zero_grad()
            loss_object = net(fixed_img, moving_img)
            loss_object.all_loss.backward()
            optimizer.step()
            all_loss_total +=loss_object.all_loss.item()
            sim_loss_total +=loss_object.similarity_loss.item()
            reg_loss_total +=loss_object.bending_energy_loss.item()
        # Record losses
        all_loss.append(all_loss_total/len(train_loader))
        sim_loss.append(sim_loss_total/len(train_loader))
        reg_loss.append(reg_loss_total/len(train_loader))

        # Periodic logging
        print(f"[{epoch}] Total: {all_loss[-1]:.4f}, "
                f"Sim: {sim_loss[-1]:.4f}, "
                f"Reg: {reg_loss[-1]:.4f}")

        # Save checkpoint
        if epoch % config.training.save_every == 0 and epoch > 0:
            checkpoint_path = Path(config.output.checkpoint_dir) / f'net_3d_{epoch}.pth'
            torch.save(net.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_path = Path(config.output.checkpoint_dir) / 'net_3d_final.pth'
    torch.save(net.state_dict(), final_path)
    print(f"Final model saved: {final_path}")

    return {
        'all_loss': all_loss,
        'sim_loss': sim_loss,
        'reg_loss': reg_loss
    }


def main(config):
    """Main function."""

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Train model
    try:
        loss_history = train_3d(config)
        print("Training completed successfully!")

        # Plot losses if matplotlib available
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].plot(loss_history['all_loss'])
            axes[0].set_title('Total Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].grid(True)

            axes[1].plot(loss_history['sim_loss'])
            axes[1].set_title('Similarity Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].grid(True)

            axes[2].plot(loss_history['reg_loss'])
            axes[2].set_title('Regularization Loss')
            axes[2].set_xlabel('Epoch')
            axes[2].grid(True)

            plt.tight_layout()
            plt.savefig(Path(config.output.results_dir) / 'training_losses_3d.png')
            print(f"Loss plot saved to: {config.output.results_dir}/training_losses_3d.png")

        except ImportError:
            print("matplotlib not available, skipping loss plot")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train 3D diffusion registration network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )
    parser.add_argument('--config', type=str, default='config/config_3d.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--data-root', type=str, default = '/mnt/c/Users/IMAG2/Documents/MATTEO/Data',
                        help='Override data root directory')
    parser.add_argument('--train-dset', type=str, default = './splits/dset_train.csv',
                        help='Override data root directory')
    parser.add_argument('--val-dset', type=str, default = './splits/dset_val.csv',
                        help='Override data root directory')
    parser.add_argument('--test-dset', type=str, default = './splits/dset_test.csv',
                        help='Override data root directory')
    parser.add_argument('--num-workers', type=int,default=4,
                        help='Number of workers')
    parser.add_argument('--epochs', type=int,default=100,
                        help='Override number of epochs')
    parser.add_argument('--lambda-reg', type=float, default=0.5,
                        help='Regularization weight')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Override learning rate')
    parser.add_argument('--loss-type', type=str, default='NewLNCC3D',
                        choices=['NewLNCC3D', 'LNCC'], help='Loss function type')
    args = parser.parse_args()

    # Load configuration
    configuration = Config(args.config)

    # Override config with command line arguments
    configuration.training.epochs = args.epochs
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
    if args.num_workers:
        configuration.training.num_workers = args.num_workers
    if args.learning_rate:
        configuration.training.learning_rate = args.learning_rate
    configuration.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for k1 in configuration.__dict__.keys():
    #     print(f'{k1}')
    #     sub_dict = getattr(configuration, k1)
    #     for k2 in sub_dict.__dict__.keys():
    #         print(f'\t{k2}: {sub_dict.__dict__[k2]}')

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
