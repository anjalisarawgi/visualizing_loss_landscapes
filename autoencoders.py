
# import umap
# from sklearn.manifold import TSNE
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import normalize
# from sklearn.manifold import trustworthiness
# from sklearn.metrics import mean_squared_error
# from sklearn.neighbors import NearestNeighbors
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, random_split, Subset
# from model import MyAwesomeModel
# from evaluation_metrics import evaluate_trustworthiness, evaluate_continuity, evaluate_mse, set_seed

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Dataset
# transform = transforms.Compose([transforms.ToTensor()])
# mnist_dataset = datasets.MNIST(
#     root='./data', train=True, transform=transform, download=True)

# train_dataset, val_dataset = random_split(mnist_dataset, [50000, 10000])
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# # Training
# net = MyAwesomeModel().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001)

# weights = []
# losses = []

# for epoch in range(5):  # Train for 5 epochs
#     net.train()
#     for data, target in train_loader:
#         data, target = data.to(device), target.to(device)  # Move data to GPU
#         optimizer.zero_grad()
#         outputs = net(data)
#         loss = criterion(outputs, target)
#         loss.backward()
#         optimizer.step()

#         # Record the weights of the first convolutional layer and the loss
#         weights.append(net.conv1.weight.data.cpu().numpy().flatten())
#         losses.append(loss.item())

# weights = np.array(weights)
# losses = np.array(losses)

# # Normalize the weights


# def l2_normalize(weights):
#     norms = np.linalg.norm(weights, axis=1, keepdims=True)
#     normalized_weights = weights / norms
#     return normalized_weights


# normalized_weights = l2_normalize(weights)


# def plot_loss_landscape(weights_reduced, method_name):
#     grid_size = 30
#     x = np.linspace(weights_reduced[:, 0].min(),
#                     weights_reduced[:, 0].max(), grid_size)
#     y = np.linspace(weights_reduced[:, 1].min(),
#                     weights_reduced[:, 1].max(), grid_size)
#     xx, yy = np.meshgrid(x, y)

#     grid_points = np.c_[xx.ravel(), yy.ravel()]
#     grid_points_orig = autoencoder.decoder(torch.tensor(
#         grid_points, dtype=torch.float32).to(device)).cpu().detach().numpy()

#     grid_losses = []
#     for i in range(grid_points_orig.shape[0]):
#         projected_weight = grid_points_orig[i].reshape(
#             net.conv1.weight.data.shape)
#         net.conv1.weight.data = torch.tensor(
#             projected_weight, dtype=torch.float32).to(device)

#         batch_losses = []
#         for data, target in val_loader:
#             data, target = data.to(device), target.to(device)
#             outputs = net(data)
#             loss = criterion(outputs, target).item()
#             batch_losses.append(loss)

#         grid_losses.append(np.mean(batch_losses))

#     grid_losses = np.array(grid_losses).reshape(xx.shape)
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(xx, yy, grid_losses, cmap='viridis', edgecolor='none')
#     ax.contour(xx, yy, grid_losses, zdir='z',
#                offset=np.min(grid_losses), cmap='viridis')
#     ax.set_xlabel(f'{method_name} Dimension 1')
#     ax.set_ylabel(f'{method_name} Dimension 2')
#     ax.set_zlabel('Loss')
#     ax.set_title(f'3D {method_name} Visualization of Loss Landscape')

#     plt.savefig(f'loss_landscape_{method_name.lower()}.png')
#     plt.close(fig)


# # Define AutoEncoder model
# class AutoEncoder(nn.Module):
#     def __init__(self, input_dim, latent_dim):
#         super(AutoEncoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, latent_dim)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, input_dim)
#         )

#     def forward(self, x):
#         latent = self.encoder(x)
#         reconstructed = self.decoder(latent)
#         return reconstructed, latent


# # Set dimensions for the autoencoder
# input_dim = normalized_weights.shape[1]
# latent_dim = 2

# # Initialize the autoencoder and move to GPU
# autoencoder = AutoEncoder(input_dim, latent_dim).to(device)
# criterion_ae = nn.MSELoss()
# optimizer_ae = optim.SGD(autoencoder.parameters(), lr=0.01)

# weights_tensor = torch.tensor(
#     normalized_weights, dtype=torch.float32).to(device)

# # Train the autoencoder with data on GPU
# num_epochs = 100
# for epoch in range(num_epochs):
#     autoencoder.train()
#     optimizer_ae.zero_grad()
#     reconstructed, latent = autoencoder(weights_tensor)
#     loss_ae = criterion_ae(reconstructed, weights_tensor)
#     loss_ae.backward()
#     optimizer_ae.step()

#     if epoch % 10 == 0:
#         print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss_ae.item():.4f}')

# # Get the latent representation of the weights
# autoencoder.eval()
# with torch.no_grad():
#     reconstructed_ae, weights_ae = autoencoder(weights_tensor)

# weights_ae = weights_ae.cpu().numpy()
# reconstructed_ae = reconstructed_ae.cpu().numpy()

# trust_ae = evaluate_trustworthiness(normalized_weights, weights_ae)
# continuity_ae = evaluate_continuity(normalized_weights, weights_ae)
# mse_ae = evaluate_mse(normalized_weights, reconstructed_ae)
# print(f'Trustworthiness of Autoencoder projection: {trust_ae:.2f}')
# print(f'Continuity of Autoencoder projection: {continuity_ae:.2f}')
# print(f'MSE of Autoencoder projection: {mse_ae:.4f}')

# print("plotting loss landscape Autoencoder")
# plot_loss_landscape(weights_ae, "Autoencoder")

# print("all done!")
