import PreProcessing
import sys
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt


# Hyperparameters etc
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 5e-5
BATCH_SIZE = 32
CHANNELS_IMG = 1
Z_DIM = 10000
NUM_EPOCHS = 1
FEATURES_CRITIC = 546
FEATURES_GEN = 546
CRITIC_ITERATIONS = 1
WEIGHT_CLIP = 0.01

def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

# custom weights initialization called on ``netG`` and ``netD``
def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            #input image of size 546 * 199
            nn.Conv2d(channels_img, features_d, kernel_size=[3,4], stride=2, padding=1),
            nn.LeakyReLU(0.2),# img: 273x100
            self._block(features_d, features_d * 2, 4, 2, 1),  # img: 136x50
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # img: 68x25
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # img: 34x12
            self._block(features_d * 8, features_d * 16, 4, 2, 1),  # img: 17x6
            self._block(features_d * 16, features_d * 32, [3,4], [1,2], 1),  # img: 8x6
            self._block(features_d * 32, features_d * 64, 4, 2, 1),  # img: 4x3
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 64, 1, kernel_size=[3,4], stride=[3,4], padding=0),
        )


    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 128, [3,4], 1, 0),  # img: 4x3
            self._block(features_g * 128, features_g * 64, [1,4], [1,2], [0,1]),  # img: 8x3
            self._block(features_g * 64, features_g * 32, [4,5], 2, 1),  # img: 17x6
            self._block(features_g * 32, features_g * 16, 4, 2, 1),  # img: 34x12
            self._block(features_g * 16, features_g * 8, [5,4], 2, 1),  # img: 68x25
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 136x50
            self._block(features_g * 4, features_g * 2, [4,5], 2, 1),  # img: 273x100
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=[3,4], stride=2, padding=1
            ),
            # Output: N x channels_img x 546 x 199
            nn.Tanh(),
        )


    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


