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
        self.hidden_to_sigma = nn.Linear(h_dims[-1], z_dim)

        # Decoder
        z_to_hidden = nn.Linear(z_dim + y_dim, h_dims[-1])
        hidden_to_input = nn.Linear(h_dims[0], x_dim)

        decoder_layers = [z_to_hidden, nn.ReLU()]
        for i, j in zip(h_dims, h_dims[1:]):
            decoder_layers.extend([nn.Linear(j, i), nn.ReLU()])
        decoder_layers.append(hidden_to_input)

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        h = self.encoder(x)

        return self.hidden_to_mu(h), self.hidden_to_sigma(h)

    def decode(self, z, c=None, activation=None):
        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        x = self.decoder(z)

        if activation == "sigmoid":
            return torch.sigmoid(x)

        return x

    def forward(self, x, c=None):
        mu, sigma = self.encode(x, c)

        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + sigma * epsilon
        x_reconstructed = self.decode(z_reparameterized, c)

        return x_reconstructed, mu, sigma


class RecurrentVAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, context_len, y_dim, dropout):
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

    def sample(self, n_samples):
        samples = torch.randn(n_samples, self.z_dim)
        decoded = self.decode(samples).view((n_samples, self.y_dim, self.context_len))
        return decoded.detach().cpu().numpy()
