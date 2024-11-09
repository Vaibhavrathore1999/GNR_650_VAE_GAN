import torch
from models import VAE_GAN  # Import your model definition

# Set the device for inference
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

# Initialize the model and load the saved parameters
gen = VAE_GAN().to(device)
gen.load_state_dict(torch.load('gen_model.pkl', map_location=device))
gen.eval()  # Set the model to evaluation mode

# Define the inference function
def generate_images(z, num_images=64):
    with torch.no_grad():  # No need to compute gradients during inference
        generated_images = gen.decoder(z.to(device))
    return generated_images

# Generate a sample of 64 images
z_sample = torch.randn((64, 128)).to(device)  # Assuming the latent dimension is 128
generated_images = generate_images(z_sample)

# Display or save the generated images
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from utils import show_and_save  # Import your show_and_save function

show_and_save("generated_images", make_grid((generated_images * 0.5 + 0.5).cpu(), nrow=8))
