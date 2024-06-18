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
import matplotlib.cm as cm
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.io as pio

from sklearn.preprocessing import StandardScaler


class MyAwesomeModel(nn.Module):
    """My awesome model with skip connections."""

    def __init__(self) -> None:
        super().__init__()

        
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)  # (input_channels, output_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Skip connection layers (1x1 convolutions to match dimensions)
        self.skip1 = nn.Conv2d(1, 32, 1, 1)  # Match input channels of 1 to output channels of 32
        self.skip2 = nn.Conv2d(32, 64, 1, 1)  # Match input channels of 32 to output channels of 64
        self.skip3 = nn.Conv2d(64, 128, 1, 1)  # Match input channels of 64 to output channels of 128

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 10)  # Adjusted to match the flattened size after pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # First conv layer with skip connection
        x1 = torch.relu(self.conv1(x))  # Output: (32, 28, 28)
        x1 = torch.max_pool2d(x1, 2, 2)  # Output: (32, 14, 14)
        skip1 = self.skip1(x)  # Output: (32, 28, 28)
        skip1 = torch.max_pool2d(skip1, 2, 2)  # Output: (32, 14, 14)
        x1 += skip1  # Add skip connection
        x1 = torch.relu(x1)

        # Second conv layer with skip connection
        x2 = torch.relu(self.conv2(x1))  # Output: (64, 14, 14)
        x2 = torch.max_pool2d(x2, 2, 2)  # Output: (64, 7, 7)
        skip2 = self.skip2(x1)  # Output: (64, 14, 14)
        skip2 = torch.max_pool2d(skip2, 2, 2)  # Output: (64, 7, 7)
        x2 += skip2  # Add skip connection
        x2 = torch.relu(x2) 

        # Third conv layer with skip connection
        x3 = torch.relu(self.conv3(x2))  # Output: (128, 7, 7)
        x3 = torch.max_pool2d(x3, 2, 2)  # Output: (128, 3, 3)
        skip3 = self.skip3(x2)  # Output: (128, 7, 7)
        skip3 = torch.max_pool2d(skip3, 2, 2)  # Output: (128, 3, 3)
        x3 += skip3  # Add skip connection
        x3 = torch.relu(x3)

        x3 = torch.flatten(x3, 1)  # Flatten the tensor
        x3 = self.dropout(x3)
        x3 = self.fc1(x3)
        return x3

# class MyAwesomeModel(nn.Module):
#     """My awesome model."""

#     def __init__(self) -> None:
#         super().__init__()

#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.conv3 = nn.Conv2d(64, 128, 3, 1)
#         self.dropout = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(128, 10)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass."""
#         x = torch.relu(self.conv1(x))
#         x = torch.max_pool2d(x, 2, 2)
#         x = torch.relu(self.conv2(x))
#         x = torch.max_pool2d(x, 2, 2)
#         x = torch.relu(self.conv3(x))
#         x = torch.max_pool2d(x, 2, 2)
#         x = torch.flatten(x, 1)
#         x = self.dropout(x)
#         x = self.fc1(x)
#         return x

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
# transform = transforms.Compose([transforms.ToTensor()])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std
])
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
print("model:", net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum = 0.9)


for epoch in range(100):
    net.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        # print("loss", loss)
        # Record the weights of the first convolutional layer and the loss
        weights.append(net.conv1.weight.data.cpu().numpy().flatten())
        losses.append(loss.item())

weights = np.array(weights)
losses = np.array(losses)
print("losses", losses.max(), losses.min())
print("losses.shape", losses.shape)
# Sample indices: one every 100 iterations
sample_indices = list(range(0, len(losses), 2000))
# Sampled weights and losses
# sample_indices = [2000, 2500, 3000, 3500, 3909] 
sampled_weights = weights[sample_indices]
sampled_losses = losses[sample_indices]

# Get the final weights and loss after training
final_weights = net.conv1.weight.data.cpu().numpy().flatten()
final_loss = losses[-1]


# Function to calculate the loss for the final weights

normalized_weights= weights

print("changes saved...")
def plot_loss_landscape(weights_reduced, method_name, sampled_weights, sampled_losses):
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
    print("grid_losses::", grid_losses.max(), grid_losses.min())
    print("grid_losses.shape::", grid_losses.shape)
    # print("grid_accuracies")
    # Create color map for different shades of red
    color_map = cm.Reds(np.linspace(0.3, 1, 11))  # Lighter to darker red
    colors = [
        (c[0], c[1], c[2], c[3]) for c in color_map
    ]

    # Plotly 3D plot for loss landscape
    surface_loss = go.Surface(z=grid_losses, x=xx, y=yy, colorscale='Viridis')
    scatter_loss = [go.Scatter3d(
        x=[sampled_weights[i, 0]],
        y=[sampled_weights[i, 1]],
        z=[sampled_losses[i]],
        mode='markers',
        marker=dict(size=5, color=colors[i % len(colors)], opacity=0.8),
        name=f'Sampled Weights and Loss {i+1}' if i == 0 else ""
    ) for i in range(len(sampled_weights))]

    fig_loss = go.Figure(data=[surface_loss] + scatter_loss)
    fig_loss.update_layout(
        title=f'3D {method_name} Visualization of Loss Landscape',
        scene=dict(
            xaxis_title=f'{method_name} Dimension 1',
            yaxis_title=f'{method_name} Dimension 2',
            zaxis_title='Loss'
        )
    )

    pio.write_html(
        fig_loss, file=f'landscape_loss_{method_name}.html', auto_open=True)

    # Plotly 2D contour plot for loss landscape
    contour_loss = go.Contour(z=grid_losses, x=x, y=y, colorscale='Viridis')
    scatter_loss_2d = [go.Scatter(
        x=[sampled_weights[i, 0]],
        y=[sampled_weights[i, 1]],
        mode='markers',
        marker=dict(size=10, color=colors[i % len(colors)], opacity=0.8),
        name=f'Sampled Weights and Loss {i+1}' if i == 0 else ""
    ) for i in range(len(sampled_weights))]

    fig_loss_2d = go.Figure(data=[contour_loss] + scatter_loss_2d)
    fig_loss_2d.update_layout(
        title=f'{method_name} Contour Plot of Loss Landscape',
        xaxis_title=f'{method_name} Dimension 1',
        yaxis_title=f'{method_name} Dimension 2'
    )

    pio.write_html(
        fig_loss_2d, file=f'landscape_loss_2d_{method_name}.html', auto_open=True)

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

    # for i in range(len(sampled_weights)):
    #     ax.scatter(sampled_weights[i, 0], sampled_weights[i, 1], sampled_losses[i],
    #                c=[colors[i % len(colors)]], s=100, label='Sampled Weights and Losses' if i == 0 else "")
    # ax.legend()

    # Contour Plot for loss landscape
    ax_contour = fig.add_subplot(2, 2, 2)
    contour = ax_contour.contourf(xx, yy, grid_losses, cmap='viridis')
    fig.colorbar(contour, ax=ax_contour)
    # for i in range(len(sampled_weights)):
    #     ax_contour.scatter(sampled_weights[i, 0], sampled_weights[i, 1], color=colors[i % len(colors)],
    #                        s=100, label='Sampled Weights and Losses' if i == 0 else "")
    ax_contour.set_xlabel(f'{method_name} Dimension 1')
    ax_contour.set_ylabel(f'{method_name} Dimension 2')
    ax_contour.set_title(f'{method_name} Contour Plot of Loss Landscape')
    # ax_contour.legend()

    # 3D Surface Plot of Accuracy Landscape
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.plot_surface(xx, yy, grid_accuracies, cmap='viridis', edgecolor='none')
    ax3.set_xlabel(f'{method_name} Dimension 1')
    ax3.set_ylabel(f'{method_name} Dimension 2')
    ax3.set_zlabel('Accuracy')
    ax3.set_title(f'3D {method_name} Visualization of Accuracy Landscape')

    # Contour Plot for Accuracy
    ax_acc = fig.add_subplot(2, 2, 4)
    contour_acc = ax_acc.contourf(xx, yy, grid_accuracies, cmap='viridis')
    fig.colorbar(contour_acc, ax=ax_acc)
    ax_acc.set_xlabel(f'{method_name} Dimension 1')
    ax_acc.set_ylabel(f'{method_name} Dimension 2')
    ax_acc.set_title(f'{method_name} Contour Plot of Accuracy Landscape')
    ax_acc.legend()

    plt.tight_layout()
    plt.savefig(f'landscape_plots_no_skip_{method_name}.png')
    plt.close(fig)


# PCA
pca = PCA(n_components=2)
weights_pca = pca.fit_transform(normalized_weights)
reconstructed_pca = pca.inverse_transform(weights_pca)

# # Calculate the 2D projection of the final weights
# optimal_weights_2d = pca.transform(
#     l2_normalize(final_weights.reshape(1, -1)))[0]

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

# # Calculate the 2D projection of the final weights for each method
sampled_weights_pca = pca.transform(sampled_weights)
sampled_weights_tsne = tsne.embedding_[sample_indices]
sampled_weights_umap = umap_reducer.transform(sampled_weights)
sampled_losses = losses[sample_indices]

print("plotting loss landscape PCA")
plot_loss_landscape(weights_pca, "PCA", sampled_weights_pca, sampled_losses)

print(f"Plotting loss landscape t-SNE ")
plot_loss_landscape(weights_tsne, "t-SNE", sampled_weights_tsne, sampled_losses)

print("plotting loss landscape UMAP")
plot_loss_landscape(weights_umap, "UMAP", sampled_weights_umap, sampled_losses)

print("all done!")