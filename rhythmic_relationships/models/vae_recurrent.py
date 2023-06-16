import torch
import torch.nn as nn


class RecurrentVAE(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim, context_len, y_dim, dropout):
        super().__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.context_len = context_len
        self.y_dim = y_dim

        # Encoder
        self.input_to_hidden = nn.RNN(
            input_size=context_len,
            hidden_size=h_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
            nonlinearity="relu",
        )

        # h_dim * 2 because of bidirectional
        self.hidden_to_mu = nn.Linear(h_dim * 2, z_dim)
        self.hidden_to_sigma = nn.Linear(h_dim * 2, z_dim)

        # Decoder
        self.z_to_h0 = nn.Sequential(nn.Linear(z_dim, h_dim), nn.Tanh())
        self.z_to_hidden = nn.RNN(
            input_size=z_dim,
            hidden_size=h_dim,
            batch_first=True,
            bidirectional=False,
            nonlinearity="relu",
        )
        # TODO: should hidden size of the second layer be context_len instead of x_dim? That would match encoder input size
        self.hidden_to_input = nn.RNN(
            input_size=h_dim,
            hidden_size=x_dim,
            batch_first=True,
            bidirectional=False,
            nonlinearity="relu",
        )
        # TODO: verify that -1 the correct dim for softmax
        self.softmax = nn.Softmax(dim=-1)

    def encode(self, x):
        unused_output, hidden = self.input_to_hidden(x)
        h = torch.cat((hidden[0], hidden[1]), dim=-1)
        return self.hidden_to_mu(h), self.hidden_to_sigma(h)

    def decode(self, z):
        # TODO: How to configure z_to_h0 given that z_to_hidden expects h0 to be of size (1, 1, 128)
        # h0 = self.z_to_h0(z)
        # output, h = self.z_to_hidden(z, h0)
        output, h = self.z_to_hidden(z)
        xoutput, xh = self.hidden_to_input(output)
        return self.softmax(xoutput)

    def forward(self, x):
        mu, sigma = self.encode(x)

        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + sigma * epsilon
        x_reconstructed = self.decode(z_reparameterized)

        return x_reconstructed, mu, sigma
