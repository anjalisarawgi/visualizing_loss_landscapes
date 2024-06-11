import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.manifold import trustworthiness
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
import umap
from mpl_toolkits.mplot3d import Axes3D

# Set the seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed = 42
set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate synthetic polynomial data
np.random.seed(42)
X = np.linspace(-1, 1, 200)
y = 5 * X**3 + 2 * X**2 + X + np.random.randn(*X.shape) * 0.2  # Polynomial with noise

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors and reshape
X_train = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_val = torch.tensor(X_val, dtype=torch.float32).reshape(-1, 1)
y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

# Create DataLoader for synthetic dataset
batch_size = 16
train_data = torch.utils.data.TensorDataset(X_train, y_train)
val_data = torch.utils.data.TensorDataset(X_val, y_val)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

# Define a simple MLP without skip connections
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Define an MLP with skip connections
class SkipMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SkipMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.skip = nn.Linear(input_size, output_size)

    def forward(self, x):
        identity = self.skip(x)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out += identity
        return out

# Training function
def train_model(model, optimizer, criterion, train_loader, epochs):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
        print(f'Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader)}')
    return train_losses

# Hyperparameters
input_size = 1
hidden_size = 64
output_size = 1
epochs = 100
learning_rate = 0.001

# Initialize models, optimizers, and loss function
model_no_skip = SimpleMLP(input_size, hidden_size, output_size).to(device)
optimizer_no_skip = optim.Adam(model_no_skip.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

model_skip = SkipMLP(input_size, hidden_size, output_size).to(device)
optimizer_skip = optim.Adam(model_skip.parameters(), lr=learning_rate)

# Train models
print("Training SimpleMLP (without skip connections)")
losses_no_skip = train_model(model_no_skip, optimizer_no_skip, criterion, train_loader, epochs)

print("Training SkipMLP (with skip connections)")
losses_skip = train_model(model_skip, optimizer_skip, criterion, train_loader, epochs)

# Plot the loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), losses_no_skip, label='Without Skip Connections')
plt.plot(range(1, epochs+1), losses_skip, label='With Skip Connections')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)
plt.savefig("polynomial.png")

# Normalize the weights
def l2_normalize(weights):
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    normalized_weights = weights / norms
    return normalized_weights

# Generate weights from trained models for visualization
def extract_weights(model):
    weights = []
    for param in model.parameters():
        if param.requires_grad:
            weights.append(param.data.cpu().numpy().flatten())
    # Concatenate all weights into a single 2D array
    return np.concatenate(weights).reshape(1, -1)

weights_no_skip = extract_weights(model_no_skip)
weights_skip = extract_weights(model_skip)

# Normalize the weights
normalized_weights_no_skip = l2_normalize(weights_no_skip)
normalized_weights_skip = l2_normalize(weights_skip)

# Function to evaluate trustworthiness
def evaluate_trustworthiness(original_data, reduced_data, n_neighbors=5):
    return trustworthiness(original_data, reduced_data, n_neighbors=n_neighbors)

# Function to evaluate continuity
def evaluate_continuity(original_data, reduced_data, n_neighbors=5):
    nbrs_original = NearestNeighbors(n_neighbors=n_neighbors).fit(original_data)
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

# Function to evaluate MSE
def evaluate_mse(original_data, reconstructed_data):
    return mean_squared_error(original_data, reconstructed_data)

# Plotting loss landscapes using PCA, t-SNE, and UMAP
def plot_loss_landscape(weights_reduced, losses, method_name):
    grid_size = 30
    x = np.linspace(weights_reduced[:, 0].min(), weights_reduced[:, 0].max(), grid_size)
    y = np.linspace(weights_reduced[:, 1].min(), weights_reduced[:, 1].max(), grid_size)
    xx, yy = np.meshgrid(x, y)

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    grid_losses = np.array(losses).reshape(xx.shape)

    fig = plt.figure(figsize=(20, 16))
    
    # 3D plot for loss landscape
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot_surface(xx, yy, grid_losses, cmap='viridis', edgecolor='none')
    ax.contour(xx, yy, grid_losses, zdir='z', offset=np.min(grid_losses), cmap='viridis')
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

    plt.tight_layout()
    plt.savefig(f'landscape_plots_{method_name}.png')
    plt.close(fig)

# Apply dimensionality reduction and plot loss landscapes
# PCA
pca = PCA(n_components=2)
weights_pca_no_skip = pca.fit_transform(normalized_weights_no_skip)
weights_pca_skip = pca.fit_transform(normalized_weights_skip)

print(f"Plotting loss landscape PCA for model without skip connections")
plot_loss_landscape(weights_pca_no_skip, losses_no_skip, "PCA_no_skip")

print(f"Plotting loss landscape PCA for model with skip connections")
plot_loss_landscape(weights_pca_skip, losses_skip, "PCA_skip")

# TSNE
tsne = TSNE(n_components=2, random_state=42)
weights_tsne_no_skip = tsne.fit_transform(normalized_weights_no_skip)
weights_tsne_skip = tsne.fit_transform(normalized_weights_skip)

print(f"Plotting loss landscape t-SNE for model without skip connections")
plot_loss_landscape(weights_tsne_no_skip, losses_no_skip, "t-SNE_no_skip")

print(f"Plotting loss landscape t-SNE for model with skip connections")
plot_loss_landscape(weights_tsne_skip, losses_skip, "t-SNE_skip")

# UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
weights_umap_no_skip = umap_reducer.fit_transform(normalized_weights_no_skip)
weights_umap_skip = umap_reducer.fit_transform(normalized_weights_skip)

print(f"Plotting loss landscape UMAP for model without skip connections")
plot_loss_landscape(weights_umap_no_skip, losses_no_skip, "UMAP_no_skip")

print(f"Plotting loss landscape UMAP for model with skip connections")
plot_loss_landscape(weights_umap_skip, losses_skip, "UMAP_skip")

print("All done!")
