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


validation_data = datasets.CIFAR10(root="data", train=False, download=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                   ]))

validation_loader = DataLoader(validation_data,
                               batch_size=32,
                               shuffle=True,
                               pin_memory=True)

model = VQVAEModel(num_hiddens, num_residual_layers, num_residual_hiddens,
                   num_embeddings, embedding_dim,
                   commitment_cost, decay).to(device)
model.load_state_dict(torch.load("model_weights/dqvae_weights.pt"))
model.eval()

(valid_originals, _) = next(iter(validation_loader))
valid_originals = valid_originals.to(device)

vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
_, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
valid_reconstructions = model._decoder(valid_quantize)


# def show(img):
#     npimg = img.numpy()
#     fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
#     fig.axes.get_xaxis().set_visible(False)
#     fig.axes.get_yaxis().set_visible(False)
#     plt.show()
#
#
# show(make_grid(valid_reconstructions.cpu().data)+0.5)
# show(make_grid(valid_originals.cpu()+0.5))

fig, axs = plt.subplots(2, 1)

axs[0].imshow(
    np.transpose(
        (make_grid(valid_reconstructions.cpu().data)+0.5).numpy(),
        (1,2,0)), interpolation='nearest'
)
axs[0].get_xaxis().set_visible(False)
axs[0].get_yaxis().set_visible(False)

axs[1].imshow(
    np.transpose(
        make_grid(valid_originals.cpu()+0.5).numpy(),
        (1,2,0)), interpolation='nearest'
)
axs[1].get_xaxis().set_visible(False)
axs[1].get_yaxis().set_visible(False)

plt.show()
