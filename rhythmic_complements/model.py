import torch
import torch.nn as nn


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim, z_dim, conditional=False, n_labels=0):
        super().__init__()

        self.conditional = conditional

        self.h_dim = h_dim
        self.z_dim = z_dim

        self.relu = nn.ReLU()

        # Encoder
        self.input_to_hidden = nn.Linear(input_dim + n_labels, h_dim)
        self.hidden_to_mu = nn.Linear(h_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(h_dim, z_dim)

        # Decoder
        self.z_to_hidden = nn.Linear(z_dim + n_labels, h_dim)
        self.hidden_to_input = nn.Linear(h_dim, input_dim)

    def encode(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        h = self.relu(self.input_to_hidden(x))
        return self.hidden_to_mu(h), self.hidden_to_sigma(h)

    def decode(self, z, c=None):
        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        h = self.relu(self.z_to_hidden(z))
        return torch.sigmoid(self.hidden_to_input(h))

    def forward(self, x, c=None):
        mu, sigma = self.encode(x, c)

        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + sigma * epsilon

        x_reconstructed = self.decode(z_reparameterized, c)

        return x_reconstructed, mu, sigma

    def loss_function(self, recons, x, mu, sigma):
        recons_loss = nn.functional.mse_loss(recons, x)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + sigma - mu**2 - sigma.exp(), dim=1), dim=0
        )
        return recons_loss + kld_loss

    def sample(self, n_samples, device):
        z = torch.randn((n_samples, self.z_dim)).to(device)
        return self.decode(z)
