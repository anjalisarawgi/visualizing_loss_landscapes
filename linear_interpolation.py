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

# set device to mps
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def l2_normalize(weights):
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    normalized_weights = weights / norms
    return normalized_weights


transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(
    root='./data', train=True, transform=transform, download=True)

subset_indices = list(range(500))  # Use only 500 samples
mnist_subset = Subset(mnist_dataset, subset_indices)
train_dataset, val_dataset = random_split(mnist_subset, [400, 100])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

net = MyAwesomeModel()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

initial_weights = None
final_weights = None

# Training loop
for epoch in range(10):  # Train for 5 epochs
    net.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        if initial_weights is None:  # Record the weights of the first convolutional layer and the loss
            initial_weights = net.conv1.weight.data.cpu().numpy().flatten()
            print("initial_weights", initial_weights.shape)
        final_weights = net.conv1.weight.data.cpu().numpy().flatten()
        # print("final_weights shape", final_weights.shape)


def plot_linear_interpolation(initial_weights, final_weights, steps=100):
    alpha_vals = np.linspace(0, 1, steps)
    interp_losses = []
    for alpha in alpha_vals:
        interpolated_weights = (1 - alpha) * \
            initial_weights + alpha * final_weights
        net.conv1.weight.data = torch.tensor(interpolated_weights.reshape(
            net.conv1.weight.data.shape), dtype=torch.float32).to(device)

        # Forward pass and loss computation
        batch_losses = []
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            loss = criterion(outputs, target).item()
            batch_losses.append(loss)

        interp_losses.append(np.mean(batch_losses))

    plt.figure(figsize=(12, 8))
    plt.plot(alpha_vals, interp_losses, marker='o', linestyle='-',
             markersize=5, linewidth=2, label='Interpolation Loss')
    plt.xlabel('Interpolation factor (α)')
    plt.ylabel('Loss')
    plt.title('Loss along Linear Interpolation Path')
    plt.savefig('loss_interpolation_sgd.png')


print("plotting loss landscape")
plot_linear_interpolation(initial_weights, final_weights)
print("done")
