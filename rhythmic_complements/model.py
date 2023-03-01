import torch
import torch.nn as nn


class VariationalAutoEncoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, conditional=False, y_dim=0):
        super().__init__()

        self.conditional = conditional

        self.h_dim = h_dim
        self.z_dim = z_dim

        self.relu = nn.ReLU()

        # TODO: what dim is correct? is that true for all reprs?
        self.softmax = nn.Softmax(dim=0)
        self.use_softmax = False

        # Encoder
        self.input_to_hidden = nn.Linear(x_dim + y_dim, h_dim)
        self.hidden_to_mu = nn.Linear(h_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(h_dim, z_dim)

        # Decoder
        self.z_to_hidden = nn.Linear(z_dim + y_dim, h_dim)
        self.hidden_to_input = nn.Linear(h_dim, x_dim)

    def encode(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        h = self.relu(self.input_to_hidden(x))
        return self.hidden_to_mu(h), self.hidden_to_sigma(h)

    def decode(self, z, c=None):
        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        h = self.relu(self.z_to_hidden(z))
        x = self.hidden_to_input(h)

        if self.use_softmax:
            return self.softmax(x)
        else:
            return torch.sigmoid(x)

    def forward(self, x, c=None):
        mu, sigma = self.encode(x, c)

        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + sigma * epsilon

        x_reconstructed = self.decode(z_reparameterized, c)

        return x_reconstructed, mu, sigma
