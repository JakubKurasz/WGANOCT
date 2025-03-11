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

#function to show images
def show_imgs(x, _, new_fig=True):
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.transpose(0,2).transpose(0,1) # channels as last dimension
    if new_fig:
        plt.figure(figsize=(16,12))
    plt.title(f"Label: {_}")
    plt.imshow(grid.numpy())
    plt.show()

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
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
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
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)

    
class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
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
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
    

if __name__ == '__main__': 

    #generate train and test dataloaders
    train_dl, test_dl = PreProcessing.preProcessing()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device: ', device)
    
    #z = torch.randn(2, 10000)

    # Re-initialize D, G:
    gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    disc = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)

    initialize_weights(gen)
    initialize_weights(disc)

    # Declare our loss function for our model
    criterion = nn.BCELoss()

   # initializate optimizer
    opt_gen = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # Creating fake samples
    lab_real = torch.ones(32, 1, device=device)
    lab_fake = torch.zeros(32, 1, device=device)


    # for logging:
    collect_x_gen = []
    fixed_noise = torch.randn(32, Z_DIM, device=device)
    fig = plt.figure() # keep updating this one
    #plt.ion()

    gen.train()
    disc.train()

    for epoch in range(NUM_EPOCHS): # no. of epochs
        ite = iter(train_dl)
        #print(len(dataloader))
        for i, data in enumerate(train_dl, 1408): # Iterate through our dataloader to train on all our samples

            # STEP 1: Discriminator optimization step
            x_real, _ = data
            x_real = x_real.to(device)
            noise = torch.randn(32, Z_DIM,1,1).to(device)
            fake = gen(noise).detach()

            disc_real = disc(x_real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()
            
            print("h")

            # STEP 2: Generator optimization step
            # reset accumulated gradients from previous iteration, backward step
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        # Logging for each epoch on loss functions for both D(x) and D(G(z))
            if True:
                x_gen = gen(fixed_noise)
                #show_imgs(x_gen, [], new_fig=False)
                #fig.canvas.draw()
                print('e{}.i{}/{} last mb D(x)={:.4f} D(G(z))={:.4f}'.format(
                    epoch, i, len(train_dl), loss_disc, loss_gen))
        # End of epoch

    x_gen = gen(fixed_noise)
    collect_x_gen.append(x_gen.detach().clone())

    for x_gen in collect_x_gen:
        show_imgs(x_gen,[])
        continue
        
    


