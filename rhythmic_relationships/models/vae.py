import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, x_dim, h_dims, z_dim, conditional=False, y_dim=0):
        super().__init__()

        self.conditional = conditional

        # Encoder
        input_to_hidden = nn.Linear(x_dim + y_dim, h_dims[0])

        encoder_layers = [input_to_hidden, nn.ReLU()]
        for i, j in zip(h_dims, h_dims[1:]):
            encoder_layers.extend([nn.Linear(i, j), nn.ReLU()])

        self.encoder = nn.Sequential(*encoder_layers)

        self.hidden_to_mu = nn.Linear(h_dims[-1], z_dim)
        self.hidden_to_logvar = nn.Linear(h_dims[-1], z_dim)

        # Decoder: the encoder's hidden dims in reverse
        reversed_dims = list(reversed(h_dims))
        z_to_hidden = nn.Linear(z_dim + y_dim, reversed_dims[0])
        hidden_to_input = nn.Linear(reversed_dims[-1], x_dim)

        decoder_layers = [z_to_hidden, nn.ReLU()]
        for i, j in zip(reversed_dims, reversed_dims[1:]):
            decoder_layers.extend([nn.Linear(i, j), nn.ReLU()])
        decoder_layers.append(hidden_to_input)

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        h = self.encoder(x)

        return self.hidden_to_mu(h), self.hidden_to_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, c=None, activation=None):
        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        x = self.decoder(z)

        if activation == "sigmoid":
            return torch.sigmoid(x)

        return x

    def forward(self, x, c=None):
        mu, logvar = self.encode(x, c)

        z_reparameterized = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z_reparameterized, c)

        return x_reconstructed, mu, logvar
