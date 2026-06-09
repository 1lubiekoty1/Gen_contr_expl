
import torch
import torch.nn as nn
from src.utils import save_model , load_model
import config

# A SIMPLE ENCODER THAT ENCODES A DIGIT IMAGE INTO A LATENT SPACE

class MnistAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential( # FROM 28x28 IMAGE TO LATENT_DIM
            nn.Flatten(),
            nn.Linear(28 * 28, 256), nn.ReLU(),
            nn.Linear(256, 128),      nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        self.decoder = nn.Sequential( # FROM LATENT_DIM TO 28x28 IMAGE
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128),        nn.ReLU(),
            nn.Linear(128, 256),        nn.ReLU(),
            nn.Linear(256, 28 * 28),   nn.Sigmoid(),  # pixels in [0,1]
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x): # SINCE DIMENTIONS CHANGE ACROSS LAYERS AUTOENCODER HAS TO FIND A WAY TO REPRESENT IMAGE IN LATENT SPACE
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x): # IF IMAGE IS NOT PLAUSIBLE (eg. some noise) THEN AUTOENCODER WILL ENCODE IT AND
                                       # THEN DECODER WILL TRY TO MAKE SOMETHING SENSIBLE OUT OF IT, BUT IT SHOULDNT (if image is noise)
                                       # SO WE MEASURE HOW MUCH IMAGE HAS CHANGED - IF A LOT => THEN IMAGE WAS UNLIKE ANY FROM THE DATASET
        with torch.no_grad():
            reconstructed = self.forward(x)
        return nn.functional.mse_loss(reconstructed, x)

def train_autoencoder(loader, epochs=10, latent_dim=32, device="cpu"): # SIMPLE TRAINER
    model = MnistAutoencoder(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for batch in loader:
            images = batch[0].to(device)
            recon  = model(images)
            loss   = nn.functional.mse_loss(recon, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} | Recon Loss: {loss.item():.4f}")

    return model

class ClassConditionalPlausibility:
    # Trains 10 separate autoencoders, one per digit.
    # Plausibility of a counterfactual targeting class C in measured
    # by how well the class-C autoencoder reconstructs it

    def __init__(self, latent_dim=32):
        self.digits = config.NUM_CLASSES
        self.autoencoders = {i: MnistAutoencoder(latent_dim) for i in range(self.digits)}
        



    def train_all(self, loader, device, epochs=20):
        for digit in range(self.digits):
            print(f"Training autoencoder for digit {digit}...")

            # FILTERS ONLY IMAGES BELONGING TO THIS DIGIT
            class_images = torch.cat([
                imgs[labels == digit]          # BOOLLEAN MASK IN EACH BATCH
                for imgs, labels in loader
            ])

            digit_dataset = torch.utils.data.TensorDataset(class_images)
            digit_loader  = torch.utils.data.DataLoader(
                digit_dataset, batch_size=64, shuffle=True
            )

            self.autoencoders[digit] = train_autoencoder(
                digit_loader, epochs=epochs, device=device
            )

    def save_trained_autoencoders( self ):
        for digit in range(self.digits):
            save_model( self.autoencoders[digit] , "autoencoder_for_digit_" + str(digit) )

    def load_saved_autoencoders( self ):
        for digit in range(self.digits):
            load_model( self.autoencoders[digit] , "autoencoder_for_digit_" + str(digit) )

    def plausibility_loss(self, counterfactual, target_class):
        # Returns reconstruction error under the target class autoencoder
        # This is differentiable - can be included in the optimization loop.
        ae = self.autoencoders[target_class].to(counterfactual.device)
        recon = ae(counterfactual)
        return nn.functional.mse_loss(recon, counterfactual)