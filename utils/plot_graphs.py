from trainer_utils import *
import sys

def plot_graphs(file_path):
    train_stats, val_stats, num_epochs = load_stats(file_path)
    plot_stats(train_stats, val_stats, num_epochs, 'FODFs prediction')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_stats.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    plot_graphs(file_path)    