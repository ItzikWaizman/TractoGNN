import torch
import matplotlib.pyplot as plt


def absolute_error(predictions, labels, valid_mask):
    absolute_error_phi_theta = torch.abs(predictions - labels) * 180/(torch.pi)
    absolute_error_phi_theta = absolute_error_phi_theta*valid_mask[: , :-1].unsqueeze(-1)
    return (absolute_error_phi_theta[..., 0].sum() / valid_mask.sum()), (absolute_error_phi_theta[:, :, 1].sum() / valid_mask.sum())


def calc_sphere_distance(phi_theta_predictions, phi_theta_labels, valid_mask):
    # Calculate the distance between points on the sphere using spherical coordinates
    phi1, theta1 = phi_theta_predictions[..., 0], phi_theta_predictions[..., 1]
    phi2, theta2 = phi_theta_labels[..., 0], phi_theta_labels[..., 1]

    delta_phi = phi1 - phi2
    
    # cos_psi is the dot product between the vectors (r = 1)
    cos_psi = (torch.cos(theta1)*torch.cos(theta2) + torch.sin(theta1)*torch.sin(theta2)*torch.cos(delta_phi))
    distance = torch.acos(cos_psi)

    # Convert distance from radians to degrees
    distance_degrees = distance * (180.0 / torch.pi)

    distance_degrees = distance_degrees*valid_mask[: , :-1]
    distance = distance_degrees.sum() / valid_mask.sum()
    
    return distance


def plot_stats(train_stats, val_stats, num_epochs, title):
    # Unpack statistics
    train_loss, train_phi_error, train_theta_error, train_sphere_distance = zip(*train_stats)
    val_loss, val_phi_error, val_theta_error, val_sphere_distance = zip(*val_stats)

    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(24, 6))

    # Plot Loss
    plt.subplot(1, 4, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot Phi Error
    plt.subplot(1, 4, 2)
    plt.plot(epochs, train_phi_error, label='Train Phi Error (deg)')
    plt.plot(epochs, val_phi_error, label='Validation Phi Error (deg)')
    plt.xlabel('Epochs')
    plt.ylabel('Phi Error (deg)')
    plt.title('Training and Validation Phi Error')
    plt.legend()

    # Plot Theta Error
    plt.subplot(1, 4, 3)
    plt.plot(epochs, train_theta_error, label='Train Theta Error (deg)')
    plt.plot(epochs, val_theta_error, label='Validation Theta Error (deg)')
    plt.xlabel('Epochs')
    plt.ylabel('Theta Error (deg)')
    plt.title('Training and Validation Theta Error')
    plt.legend()

    # Plot Sphere Distance
    plt.subplot(1, 4, 4)
    plt.plot(epochs, train_sphere_distance, label='Train Sphere Distance (deg)')
    plt.plot(epochs, val_sphere_distance, label='Validation Sphere Distance (deg)')
    plt.xlabel('Epochs')
    plt.ylabel('Sphere Distance (deg)')
    plt.title('Training and Validation Sphere Distance')
    plt.legend()

    # Add a big title for the entire figure
    plt.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

