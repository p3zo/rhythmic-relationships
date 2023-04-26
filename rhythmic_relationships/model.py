import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, x_dim, h_dims, z_dim, conditional=False, y_dim=0):
        super().__init__()

        self.conditional = conditional

        self.h_dims = h_dims
        self.z_dim = z_dim

        self.relu = nn.ReLU()

        # TODO: what dim is correct? is that true for all reprs?
        self.softmax = nn.Softmax(dim=0)

        # Encoder
        self.input_to_hidden = nn.Linear(x_dim + y_dim, h_dims[0])
        self.hidden_to_mu = nn.Linear(h_dims[-1], z_dim)
        self.hidden_to_sigma = nn.Linear(h_dims[-1], z_dim)

        # Decoder
        self.z_to_hidden = nn.Linear(z_dim + y_dim, h_dims[-1])
        self.hidden_to_input = nn.Linear(h_dims[0], x_dim)

        # Hidden layers for both encoder and decoder
        for i, j in zip(h_dims, h_dims[1:]):
            setattr(self, f"hidden_{i}_to_hidden_{j}", nn.Linear(i, j))
            setattr(self, f"hidden_{j}_to_hidden_{i}", nn.Linear(j, i))

    def encode(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        h = self.relu(self.input_to_hidden(x))

        for i, j in zip(self.h_dims, self.h_dims[1:]):
            h_layer = getattr(self, f"hidden_{i}_to_hidden_{j}")
            h = self.relu(h_layer(h))

        return self.hidden_to_mu(h), self.hidden_to_sigma(h)

    def decode(self, z, c=None, activation=None):
        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        h = self.relu(self.z_to_hidden(z))

        h_dims_rev = list(reversed(self.h_dims))
        for i, j in zip(h_dims_rev, h_dims_rev[1:]):
            h_layer = getattr(self, f"hidden_{i}_to_hidden_{j}")
            h = self.relu(h_layer(h))

        x = self.hidden_to_input(h)

        if activation == "softmax":
            return self.softmax(x)
        elif activation == "sigmoid":
            return torch.sigmoid(x)
        return x

    def forward(self, x, c=None):
        mu, sigma = self.encode(x, c)

        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + sigma * epsilon

        x_reconstructed = self.decode(z_reparameterized, c)

        return x_reconstructed, mu, sigma
