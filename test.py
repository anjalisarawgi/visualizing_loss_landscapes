
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

# Define utility functions
# set device to mps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


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

# Define CNN for MNIST


def l2_normalize(weights):
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    normalized_weights = weights / norms
    return normalized_weights


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 32*7*7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x


# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(
    root='./data', train=True, transform=transform, download=True)

# # # Reduce dataset size
subset_indices = list(range(50000))  # Use only 5000 samples
mnist_subset = Subset(mnist_dataset, subset_indices)
train_dataset, val_dataset = random_split(mnist_subset, [40000, 10000])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# train_dataset, val_dataset = random_split(mnist_dataset, [50000, 10000])
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize network, loss function, and optimizer
net = MyAwesomeModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Variables to store weights and losses
weights = []
losses = []

# Training loop
for epoch in range(5):  # Train for 5 epochs
    net.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        # print("outputs", outputs.shape)

        # Record the weights of the first convolutional layer and the loss
        weights.append(net.conv1.weight.data.cpu().numpy().flatten())
        losses.append(loss.item())
    # Record the weights of the first convolutional layer and the loss
    # weights.append(net.conv1.weight.data.cpu().numpy().flatten())

weights = np.array(weights)
print("weights", weights.shape)
losses = np.array(losses)
print("losses", losses.shape)
# Normalize the weights
# normalized_weights = normalize(weights, axis=1)
normalized_weights = l2_normalize(weights)

# Function to plot loss landscape


def plot_loss_landscape(weights_reduced, method_name):
    grid_size = 30
    x = np.linspace(weights_reduced[:, 0].min(),
                    weights_reduced[:, 0].max(), grid_size)
    y = np.linspace(weights_reduced[:, 1].min(),
                    weights_reduced[:, 1].max(), grid_size)
    print("x", x.shape)
    print("y", y.shape)
    xx, yy = np.meshgrid(x, y)
    print("xx", xx.shape)
    print("yy", yy.shape)

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    print("xx.ravel", xx.ravel().shape)
    print("yy.ravel", yy.ravel().shape)
    # print("grid_points", grid_points.shape)
    # For autoencoder and UMAP, use inverse transform
    if method_name == "Autoencoder":
        grid_points_orig = autoencoder.decoder(torch.tensor(
            grid_points, dtype=torch.float32)).detach().numpy()
    elif method_name == "UMAP":
        grid_points_orig = umap_reducer.inverse_transform(grid_points)
    else:
        # PCA and t-SNE do not have inverse transform, so use PCA for both
        grid_points_orig = pca.inverse_transform(grid_points)
    # print("grid_points_orig", grid_points_orig.shape)

    grid_losses = []
    for i in range(grid_points_orig.shape[0]):
        # Set the network weights to the projected weights
        projected_weight = grid_points_orig[i].reshape(
            net.conv1.weight.data.shape)
        net.conv1.weight.data = torch.tensor(
            projected_weight, dtype=torch.float32)

        # Forward pass and loss computation
        batch_losses = []
        for data, target in val_loader:
            outputs = net(data)
            loss = criterion(outputs, target).item()
            batch_losses.append(loss)

        grid_losses.append(np.mean(batch_losses))

    grid_losses = np.array(grid_losses).reshape(xx.shape)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, grid_losses, cmap='viridis', edgecolor='none')
    ax.contour(xx, yy, grid_losses, zdir='z',
               offset=np.min(grid_losses), cmap='viridis')
    ax.set_xlabel(f'{method_name} Dimension 1')
    ax.set_ylabel(f'{method_name} Dimension 2')
    ax.set_zlabel('Loss')
    ax.set_title(f'3D {method_name} Visualization of Loss Landscape')

    # Save plot instead of displaying it
    plt.savefig(f'test_loss_landscape_{method_name.lower()}.png')
    plt.close(fig)


# Apply PCA to the normalized weights
pca = PCA(n_components=2)
print("normalized_weights", normalized_weights.shape)
weights_pca = pca.fit_transform(normalized_weights)
print("weights_pca", weights_pca.shape)
reconstructed_pca = pca.inverse_transform(weights_pca)
print("reconstructed_pca", reconstructed_pca.shape)

# Evaluate the trustworthiness, continuity, and MSE of the PCA projection
trust_pca = evaluate_trustworthiness(normalized_weights, weights_pca)
continuity_pca = evaluate_continuity(normalized_weights, weights_pca)
mse_pca = evaluate_mse(normalized_weights, reconstructed_pca)
print(f'Trustworthiness of PCA projection: {trust_pca:.2f}')
print(f'Continuity of PCA projection: {continuity_pca:.2f}')
print(f'MSE of PCA projection: {mse_pca:.4f}')

# Apply t-SNE to the normalized weights
tsne = TSNE(n_components=2, random_state=42)
weights_tsne = tsne.fit_transform(normalized_weights)

# Note: t-SNE does not have an inverse transform, so MSE cannot be computed
trust_tsne = evaluate_trustworthiness(normalized_weights, weights_tsne)
continuity_tsne = evaluate_continuity(normalized_weights, weights_tsne)
print(f'Trustworthiness of t-SNE projection: {trust_tsne:.2f}')
print(f'Continuity of t-SNE projection: {continuity_tsne:.2f}')

# Define AutoEncoder model


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


# Set dimensions for the autoencoder
input_dim = normalized_weights.shape[1]
latent_dim = 2

# Initialize the autoencoder
autoencoder = AutoEncoder(input_dim, latent_dim)
criterion_ae = nn.MSELoss()
optimizer_ae = optim.Adam(autoencoder.parameters(), lr=0.01)

# Prepare the weights for autoencoder training
weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32)

# Train the autoencoder
num_epochs = 100
for epoch in range(num_epochs):
    autoencoder.train()
    optimizer_ae.zero_grad()
    reconstructed, latent = autoencoder(weights_tensor)
    loss_ae = criterion_ae(reconstructed, weights_tensor)
    loss_ae.backward()
    optimizer_ae.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss_ae.item():.4f}')

# Get the latent representation of the weights
autoencoder.eval()
with torch.no_grad():
    reconstructed_ae, weights_ae = autoencoder(weights_tensor)

weights_ae = weights_ae.numpy()
reconstructed_ae = reconstructed_ae.numpy()

# Evaluate the trustworthiness, continuity, and MSE of the Autoencoder projection
trust_ae = evaluate_trustworthiness(normalized_weights, weights_ae)
continuity_ae = evaluate_continuity(normalized_weights, weights_ae)
mse_ae = evaluate_mse(normalized_weights, reconstructed_ae)
print(f'Trustworthiness of Autoencoder projection: {trust_ae:.2f}')
print(f'Continuity of Autoencoder projection: {continuity_ae:.2f}')
print(f'MSE of Autoencoder projection: {mse_ae:.4f}')

# Apply UMAP to the normalized weights
umap_reducer = umap.UMAP(n_components=2, random_state=42)
weights_umap = umap_reducer.fit_transform(normalized_weights)

# Note: UMAP does not have an inverse transform, so MSE cannot be computed
trust_umap = evaluate_trustworthiness(normalized_weights, weights_umap)
continuity_umap = evaluate_continuity(normalized_weights, weights_umap)
print(f'Trustworthiness of UMAP projection: {trust_umap:.2f}')
print(f'Continuity of UMAP projection: {continuity_umap:.2f}')

# Plot loss landscapes for each method
print("plotting loss landscape")
# plot_loss_landscape(weights_pca, "PCA")
print("plotting loss landscape pca")
# plot_loss_landscape(weights_tsne, "t-SNE")
# plot_loss_landscape(weights_ae, "Autoencoder")
plot_loss_landscape(weights_umap, "UMAP")
