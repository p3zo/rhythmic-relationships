import torch
from tqdm import tqdm


def train(
    model,
    train_loader,
    optimizer,
    input_dim,
    device="cpu",
    num_epochs=5,
    clip_gradients=False,
    conditional=False,
):

    for epoch in range(num_epochs):
        batches = tqdm(train_loader)
        for batch in batches:
            # Forward pass
            if conditional:
                x, y = batch
                x, y = x.to(device).view(x.shape[0], input_dim), y.to(device)
                x_reconstructed, mu, sigma = model(x, y)
            else:
                x = batch
                x = x.to(device).view(x.shape[0], input_dim)
                x_reconstructed, mu, sigma = model(x)

            # Compute loss
            loss = model.loss_function(x_reconstructed, x, mu, sigma)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            batches.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            batches.set_postfix({"loss": loss.item()})
