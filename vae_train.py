# https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from utils.vae import VQVAEModel
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# params
num_training_updates = 15000
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 512
commitment_cost = 0.25

decay = 0.99
batch_size = 256
learning_rate = 1e-3


# data
training_data = datasets.CIFAR10(root="data", train=True, download=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                 ]))

validation_data = datasets.CIFAR10(root="data", train=False, download=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                   ]))
data_variance = np.var(training_data.data / 255.0)

training_loader = DataLoader(training_data,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True)


# model
model = VQVAEModel(num_hiddens, num_residual_layers, num_residual_hiddens,
                   num_embeddings, embedding_dim,
                   commitment_cost, decay).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
model.train()


# training
train_res_recon_error = []
train_res_perplexity = []

for i in range(num_training_updates):
    (data, _) = next(iter(training_loader))
    data = data.to(device)

    optimizer.zero_grad()

    vq_loss, data_recon, perplexity = model(data)
    recon_error = F.mse_loss(data_recon, data) / data_variance
    loss = recon_error + vq_loss

    loss.backward()
    optimizer.step()

    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())

    if (i + 1) % 100 == 0:
        print('%d iterations' % (i + 1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
        print()


# plot
# train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
# train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)
# f = plt.figure(figsize=(16,8))
# ax = f.add_subplot(1,2,1)
# ax.plot(train_res_recon_error_smooth)
# ax.set_yscale('log')
# ax.set_title('Smoothed NMSE.')
# ax.set_xlabel('iteration')
#
# ax = f.add_subplot(1,2,2)
# ax.plot(train_res_perplexity_smooth)
# ax.set_title('Smoothed Average codebook usage (perplexity).')
# ax.set_xlabel('iteration')
# plt.show()
#
#
# torch.save(model.state_dict(), "model_weights/dqvae_weights.pt")
