from model.datamode import DataMode
from pathlib import Path
import torch

# ----------- PROCESSOR -----------
CUDA_AVAILABLE = torch.cuda.is_available()
MPS_AVAILABLE = torch.backends.mps.is_available()
DEVICE_NAME = "cuda" if CUDA_AVAILABLE else "mps" if MPS_AVAILABLE else "cpu"
DEVICE = torch.device(DEVICE_NAME)

# ------------ DATA ------------
DATA_FILE = Path("datasets/" )
VERIF_NET = Path(f"models/verification_net")
VERIF_NET_THRESHOLD = .8
VERIF_NET_LEARNING_RATE = 1e-3
VERIF_NET_EPOCHS = 100
VERIF_NET_BATCH_SIZE = 258

# ---- AUTOENCODER HYPERPARAMETERS ---
AE_LATENT_SPACE_SIZE = 64
AE_DATAMODE = DataMode.MONO
AE_LEARNING_RATE = 1e-4
AE_BETAS = (0.9, 0.999)
AE_MODEL = Path(f"models/ae_{AE_LATENT_SPACE_SIZE}")
AE_TRAINER = Path(f"trainers/ae_{AE_LATENT_SPACE_SIZE}.pkl")

# ---- VARIATIONAL AUTOENCODER HYPERPARAMETERS ---
VAE_LATENT_SPACE_SIZE = 10
VAE_DATAMODE = DataMode.MONO | DataMode.BINARY
VAE_LEARNING_RATE = 1e-5
VAE_BETAS = (0.9, 0.999)
VAE_BETA_KL = 1.
VAE_DATAMODE = DataMode.BINARY | DataMode.COLOR
VAE_MODEL = Path(f"models/vae_{VAE_LATENT_SPACE_SIZE}")
VAE_TRAINER = Path(f"trainers/vae_{VAE_LATENT_SPACE_SIZE}.pkl")
