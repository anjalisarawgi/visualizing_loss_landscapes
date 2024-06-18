import umap
from sklearn.manifold import TSNE
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.manifold import trustworthiness
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from model import MyAwesomeModel
# from visualizing_loss_landscapes.misc.autoencoders import train_autoencoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setting the metrics for evaluation
def evaluate_trustworthiness(original_data, reduced_data, n_neighbors=5):
    return trustworthiness(original_data, reduced_data, n_neighbors=n_neighbors)


def evaluate_continuity(original_data, reduced_data, n_neighbors=5):
    nbrs_original = NearestNeighbors(
        n_neighbors=n_neighbors).fit(original_data)
    nbrs_reduced = NearestNeighbors(n_neighbors=n_neighbors).fit(reduced_data)

    indices_original = nbrs_original.kneighbors(return_distance=False)
    indices_reduced = nbrs_reduced.kneighbors(return_distance=False)

    continuity_sum = 0.0
    n_samples = original_data.shape[0]

    for i in range(n_samples):
        intersect = len(set(indices_original[i]) & set(indices_reduced[i]))
        continuity_sum += intersect / n_neighbors

    continuity_score = continuity_sum / n_samples
    return continuity_score


def evaluate_mse(original_data, reconstructed_data):
    return mean_squared_error(original_data, reconstructed_data)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed = 42
set_seed(seed)


# Dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(
    root='./data', train=True, transform=transform, download=True)

# # Reduce dataset size
# subset_indices = list(range(500))  # Use only 500 samples
# mnist_subset = Subset(mnist_dataset, subset_indices)
# train_dataset, val_dataset = random_split(mnist_subset, [400, 100])

train_dataset, val_dataset = random_split(mnist_dataset, [50000, 10000])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# Train for 5 epochs
weights = []
losses = []

net = MyAwesomeModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum = 0.9)

for epoch in range(5):
    net.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # Record the weights of the first convolutional layer and the loss
        weights.append(net.conv1.weight.data.cpu().numpy().flatten())
        losses.append(loss.item())

weights = np.array(weights)
losses = np.array(losses)


# Function to calculate the loss for the final weights

# Get the final weights after training

def l2_normalize(weights):
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    normalized_weights = weights / norms
    return normalized_weights

# normalizing the weights
normalized_weights = l2_normalize(weights)

# function to plot our loss landscapes
def plot_loss_landscape(weights_reduced, method_name):
    grid_size = 30
    x = np.linspace(weights_reduced[:, 0].min(),
                    weights_reduced[:, 0].max(), grid_size)
    y = np.linspace(weights_reduced[:, 1].min(),
                    weights_reduced[:, 1].max(), grid_size)
    xx, yy = np.meshgrid(x, y)

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    if method_name == "UMAP":
        grid_points_orig = umap_reducer.inverse_transform(grid_points)
    else:
        grid_points_orig = pca.inverse_transform(grid_points)

    grid_losses = []
    grid_accuracies = []
    for i in range(grid_points_orig.shape[0]):
        projected_weight = grid_points_orig[i].reshape(
            net.conv1.weight.data.shape)
        net.conv1.weight.data = torch.tensor(
            projected_weight, dtype=torch.float32).to(device)

        batch_losses = []
        correct = 0
        total = 0
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            loss = criterion(outputs, target).item()
            batch_losses.append(loss)
            
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()


        grid_losses.append(np.mean(batch_losses))
        grid_accuracies.append(correct / total)

    grid_losses = np.array(grid_losses).reshape(xx.shape)
    grid_accuracies = np.array(grid_accuracies).reshape(xx.shape)

    fig = plt.figure(figsize=(20, 16))
    
    # 3D plot for loss landscape
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot_surface(xx, yy, grid_losses, cmap='viridis', edgecolor='none')
    ax.contour(xx, yy, grid_losses, zdir='z',
               offset=np.min(grid_losses), cmap='viridis')
    ax.set_xlabel(f'{method_name} Dimension 1')
    ax.set_ylabel(f'{method_name} Dimension 2')
    ax.set_zlabel('Loss')
    ax.set_title(f'3D {method_name} Visualization of Loss Landscape')

    # Highlight the final weights and final loss in 3D plot
    final_weight = weights_reduced[-1]
    final_loss = losses[-1]
    ax.scatter(final_weight[0], final_weight[1], final_loss, color='r', s=100, label='Final Weights and Loss')
    ax.legend()


    # Contour Plot for loss landscape
    ax_contour = fig.add_subplot(2, 2, 2)
    contour = ax_contour.contourf(xx, yy, grid_losses, cmap='viridis')
    fig.colorbar(contour, ax=ax_contour)
    ax_contour.scatter(final_weight[0], final_weight[1], color='r', s=100, label='Final Weights and Loss')
    ax_contour.set_xlabel(f'{method_name} Dimension 1')
    ax_contour.set_ylabel(f'{method_name} Dimension 2')
    ax_contour.set_title(f'{method_name} Contour Plot of Loss Landscape')
    ax_contour.legend()

    # 3D Surface Plot of Accuracy Landscape
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.plot_surface(xx, yy, grid_accuracies, cmap='viridis', edgecolor='none')
    ax3.set_xlabel(f'{method_name} Dimension 1')
    ax3.set_ylabel(f'{method_name} Dimension 2')
    ax3.set_zlabel('Accuracy')
    ax3.set_title(f'3D {method_name} Visualization of Accuracy Landscap')

    # Contour Plot for Accuracy
    ax_acc = fig.add_subplot(2, 2, 4)
    contour_acc = ax_acc.contourf(xx, yy, grid_accuracies, cmap='viridis')
    fig.colorbar(contour_acc, ax=ax_acc)
    ax_acc.scatter(final_weight[0], final_weight[1], color='r', s=100, label='Final Weights')
    ax_acc.set_xlabel(f'{method_name} Dimension 1')
    ax_acc.set_ylabel(f'{method_name} Dimension 2')
    ax_acc.set_title(f'{method_name} Contour Plot of Accuracy Landscape')
    ax_acc.legend()

    plt.tight_layout()
    plt.savefig(f'landscape_plots_{method_name}.png')
    plt.close(fig)

# PCA
pca = PCA(n_components=2)
weights_pca = pca.fit_transform(normalized_weights)
reconstructed_pca = pca.inverse_transform(weights_pca)

trust_pca = evaluate_trustworthiness(normalized_weights, weights_pca)
continuity_pca = evaluate_continuity(normalized_weights, weights_pca)
mse_pca = evaluate_mse(normalized_weights, reconstructed_pca)
print(f'Trustworthiness of PCA projection: {trust_pca:.2f}')
print(f'Continuity of PCA projection: {continuity_pca:.2f}')
print(f'MSE of PCA projection: {mse_pca:.4f}')

# TSNE
tsne = TSNE(n_components=2, random_state=42)
weights_tsne = tsne.fit_transform(normalized_weights)
trust_tsne = evaluate_trustworthiness(normalized_weights, weights_tsne)
continuity_tsne = evaluate_continuity(normalized_weights, weights_tsne)
print(f'Trustworthiness of t-SNE projection: {trust_tsne:.2f}')
print(f'Continuity of t-SNE projection: {continuity_tsne:.2f}')

# UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
weights_umap = umap_reducer.fit_transform(normalized_weights)

trust_umap = evaluate_trustworthiness(normalized_weights, weights_umap)
continuity_umap = evaluate_continuity(normalized_weights, weights_umap)
print(f'Trustworthiness of UMAP projection: {trust_umap:.2f}')
print(f'Continuity of UMAP projection: {continuity_umap:.2f}')

# # Plot loss landscapes for each method
print("plotting loss landscape PCA")
plot_loss_landscape(weights_pca, "PCA")

print(f"Plotting loss landscape t-SNE ")
plot_loss_landscape(weights_tsne, "t-SNE")

print("plotting loss landscape UMAP")
plot_loss_landscape(weights_umap, "UMAP")

print("all done!")

