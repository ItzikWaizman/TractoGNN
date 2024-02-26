from trainers.gcn_autoencoder_trainer import GCNAETrainer
from config import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    print(f"Device: {DEVICE}")
    gt_graph = torch.load(os.path.join(TORCH_DATA_DIR, "1013", "gt_graph.pt")).to(DEVICE)
    training_graph = torch.load(os.path.join(TORCH_DATA_DIR, "1013", "training_graph.pt")).to(DEVICE)

    gcn_trainer = GCNAETrainer()
    losses = []

    for i in tqdm(range(TRAINING_EPOCHS)):
        loss = gcn_trainer.train_iteration(input_graphs=[training_graph], gt_graphs=[gt_graph])
        losses.append(loss)

    plt.figure()
    plt.plot(np.arange(TRAINING_EPOCHS), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross Entropy with Logits Loss")
    plt.title("Learning Curve: GT Graph Reconstruction Loss")
    plt.savefig("LearningCurve")
