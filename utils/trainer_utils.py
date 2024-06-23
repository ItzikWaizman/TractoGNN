import torch
import matplotlib.pyplot as plt

def plot_stats(train_stats, val_stats, num_epochs, title):
    # Unpack statistics
    train_loss, train_acc, train_acc_top_k = zip(*train_stats)
    val_loss, val_acc, val_acc_top_k = zip(*val_stats)


    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(24, 6))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot Phi Error
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_acc, label='Train Accuracy (top1)')
    plt.plot(epochs, val_acc, label='Validation Accuracy (top1)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot Theta Error
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_acc_top_k, label='Train Top k Accuracy')
    plt.plot(epochs, val_acc_top_k, label='Validation Top k Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Top k Accuracy')
    plt.title('Training and Validation Top K Accuracy')
    plt.legend()

    # Add a big title for the entire figure
    plt.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def print_model_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print(f"{name}: {num_params} parameters")
    print(f"Total number of parameters: {total_params}")


def calc_metrics(outputs, labels, valid_mask, k):
    top1_pred_indices = torch.argmax(outputs, dim=-1)
    top1_label_indices = torch.argmax(labels, dim=-1)
    top_k_label_indices = torch.topk(labels, k=k, dim=-1)[1]
    correct_top_1 = top1_pred_indices == top1_label_indices
    correct_top_k = torch.any(torch.eq(top1_pred_indices.unsqueeze(-1), top_k_label_indices), dim=-1)
    acc_top_1 = torch.sum(correct_top_1 * (valid_mask)) / (valid_mask).sum()
    acc_top_k = torch.sum(correct_top_k * (valid_mask)) / (valid_mask).sum()

    return acc_top_1.item(), acc_top_k.item()
