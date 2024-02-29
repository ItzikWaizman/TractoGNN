import os
import torch

# directory  definitions
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'Raw')
TORCH_DATA_DIR = os.path.join(DATA_DIR, 'TorchData')
DATA_HANDLING_DIR = os.path.join(ROOT_DIR, 'data_handling')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
UTILS_DIR = os.path.join(ROOT_DIR, 'utils')

# parameters
TRAIN_CONNECT_RADIUS = 1
FILTER_GT_CON = False
FILTER_TRAIN_CON = False
LR = 5e-4
TRAINING_EPOCHS = 50
FEATURE_DIMS = [100, 50, 25, 10]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
