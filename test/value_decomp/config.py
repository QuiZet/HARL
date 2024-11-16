import torch

GAMMA = 0.99
LR = 1e-3
CLIP_EPS = 0.2
EPOCHS = 3
BATCH_SIZE = 64
ENTROPY_COEFF = 0.01
MAX_GRAD_NORM = 0.5
TIMESTEPS = 100000

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using device: {device}')