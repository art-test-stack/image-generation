from model.datamode import DataMode
import torch

# ----------- PROCESSOR -----------
CUDA_AVAILABLE = torch.cuda.is_available()
MPS_AVAILABLE = torch.backends.mps.is_available()
DEVICE_NAME = "cuda" if CUDA_AVAILABLE else "mps" if MPS_AVAILABLE else "cpu"
DEVICE = torch.device(DEVICE_NAME)

# ------------ DATA ------------
DATA_FILE = "datasets/" 
VERIF_NET = f"models/verification_net"
VERIF_NET_THRESHOLD = .8
VERIF_NET_LEARNING_RATE = 1e-4
VERIF_NET_EPOCHS = 100
VERIF_NET_BATCH_SIZE = 258

# ---- AUTOENCODER HYPERPARAMETERS ---
AE_LATENT_SPACE_SIZE = 64
AE_DATAMODE = DataMode.MONO
AE_LEARNING_RATE = 1e-4
AE_BETAS = (0.9, 0.999)
AE_MODEL = f"models/ae_{AE_LATENT_SPACE_SIZE}"
AE_TRAINER = f"trainers/ae_{AE_LATENT_SPACE_SIZE}.pkl"

# ---- VARIATIONAL AUTOENCODER HYPERPARAMETERS ---
VAE_LATENT_SPACE_SIZE = 10
VAE_DATAMODE = DataMode.MONO
VAE_LEARNING_RATE = 1e-4
VAE_BETAS = (0.9, 0.999)
VAE_DATAMODE = DataMode.BINARY | DataMode.COLOR
VAE_MODEL = f"models/vae_{VAE_LATENT_SPACE_SIZE}"
VAE_TRAINER = f"trainers/vae_{VAE_LATENT_SPACE_SIZE}.pkl"