# DGIR
[Guiding Registration with Emergent Similarity from Pre-Trained Diffusion Models](https://arxiv.org/abs/2506.02419) (MICCAI 2025 Spotlight)


A PyTorch implementation of diffusion-regularized neural networks for medical image registration, supporting both 2D and 3D registration tasks with state-of-the-art performance.

### Dependencies

You'll also need the weights from [guided_diffusion](https://github.com/openai/guided-diffusion):

```bash
# Clone guided_diffusion as a submodule or install separately
cd guided_diffusion
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
```

## Quick Start

### Basic 2D Registration

```python
from diffusion_registration import Config, DiffusionRegistrationNet
from diffusion_registration.training import Trainer
from diffusion_registration.data import create_data_loaders

# Load configuration
config = Config('config/config.yaml')
config.model.dimension = 2
config.data.data_root = '/path/to/your/data'

# Create data loaders
train_dataset, test_dataset = create_data_loaders(config)

# Initialize network
net = DiffusionRegistrationNet(config)
net = net.to(config.device)

# Train
trainer = Trainer(net, config)
trainer.train(train_dataset, test_dataset)
```

### Command Line Training

```bash
# Train 2D registration
python scripts/train_2d.py --config config/config.yaml

# Train 3D registration
python scripts/train_3d.py --config config/config_3d.yaml 
```
