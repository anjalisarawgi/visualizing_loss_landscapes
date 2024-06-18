import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split, Subset

# Load MNIST data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = datasets.MNIST(
    root='./data', train=True, transform=transform, download=True)

# # Reduce dataset size
subset_indices = list(range(5000))  # Use only 500 samples
mnist_subset = Subset(mnist_dataset, subset_indices)
train_dataset, val_dataset = random_split(mnist_subset, [4000, 1000])

# train_dataset, val_dataset = random_split(mnist_dataset, [50000, 10000])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define a simple neural network


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


model = MyAwesomeModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training function


def train_model(epochs=5):
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

print("training model")
train_model()

print("model trained")
def linear_interpolation(model, criterion, point1, point2, steps=50):
    alpha_vals = np.linspace(0, 1, steps)
    losses = []

    for alpha in alpha_vals:
        interp_params = [(1 - alpha) * p1 + alpha *
                         p2 for p1, p2 in zip(point1, point2)]
        model.load_state_dict({name: torch.tensor(param) for name, param in zip(
            model.state_dict().keys(), interp_params)})

        total_loss = 0
        with torch.no_grad():
            for data, target in train_loader:
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()

        losses.append(total_loss / len(train_loader))

    plt.plot(alpha_vals, losses)
    plt.xlabel('Alpha')
    plt.ylabel('Loss')
    plt.title('Linear Interpolation of Loss')
    plt.savefig('linear_interpolation.png')


# Save initial and final parameters
initial_params = [param.clone() for param in model.parameters()]
# Perform an additional epoch of training to change the model parameters
train_model(epochs=1)
final_params = [param.clone() for param in model.parameters()]

# linear_interpolation(model, criterion, initial_params, final_params)


# 2d contour plot
print("2d contour plot")
def plot_2d_contour(model, criterion, point, direction1, direction2, steps=50):
    alpha_vals = np.linspace(-1, 1, steps)
    beta_vals = np.linspace(-1, 1, steps)
    losses = np.zeros((steps, steps))

    for i, alpha in enumerate(alpha_vals):
        for j, beta in enumerate(beta_vals):
            new_params = [p + alpha * d1 + beta * d2 for p,
                          d1, d2 in zip(point, direction1, direction2)]
            model.load_state_dict({name: torch.tensor(param) for name, param in zip(
                model.state_dict().keys(), new_params)})

            total_loss = 0
            with torch.no_grad():
                for data, target in train_loader:
                    output = model(data)
                    loss = criterion(output, target)
                    total_loss += loss.item()

            losses[i, j] = total_loss / len(train_loader)

    X, Y = np.meshgrid(alpha_vals, beta_vals)
    plt.contourf(X, Y, losses, levels=20, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.title('2D Contour Plot of Loss')
    plt.savefig("2d_contour.png")


direction1 = [torch.randn_like(p) for p in model.parameters()]
direction2 = [torch.randn_like(p) for p in model.parameters()]
# plot_2d_contour(model, criterion, initial_params, direction1, direction2)


# filter normalization
def filter_normalize(directions, reference_params):
    norm_directions = []
    for direction, param in zip(directions, reference_params):
        norm_direction = direction * \
            (torch.norm(param) / torch.norm(direction))
        norm_directions.append(norm_direction)
    return norm_directions


norm_direction1 = filter_normalize(direction1, initial_params)
norm_direction2 = filter_normalize(direction2, initial_params)
plot_2d_contour(model, criterion, initial_params,
                norm_direction1, norm_direction2)
