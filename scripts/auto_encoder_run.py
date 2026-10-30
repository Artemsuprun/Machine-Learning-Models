import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from models.auto_encoder import AutoEncoder
from datasets.general import get_datasets
from utils.device import get_device


# get the device aviable for your computer
device = get_device()

# Create the model architecture (must match the one used during training)
model = AutoEncoder(encoder_dim=[784, 128, 64], activation="relu", output_activation="sigmoid")

# Load the trained model weights from the file
model.load_state_dict(torch.load("./checkpoints/autoencoder_mnist.pth", map_location=device))
model.to(device)
# Set the model to evaluation mode
model.eval()

# apply corruption to the input images (e.g., adding noise or masking random pixels)
clean_transform = transforms.Compose([
    transforms.ToTensor(),
])
corrupt_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.35, 2.0)),
])

clean_test_set = get_datasets("mnist", clean_transform)[1]
corrupt_test_set = get_datasets("mnist", corrupt_transform)[1]

# get random image from test set and reconstruct it using the trained model
rnd_idx = random.randint(0, len(clean_test_set) - 1)
clean_img, _ = clean_test_set[rnd_idx]
corrupt_img, _ = corrupt_test_set[rnd_idx]

input_img = corrupt_img.unsqueeze(0).to(device)

with torch.no_grad():
    reconstructed_img = model(input_img)

# move the reconstructed image back to cpu to plot it.
reconstructed_img = reconstructed_img.detach().squeeze().cpu()

# Plot original and reconstructed images side by side
plt.figure(figsize=(8, 4))

plt.subplot(1, 3, 1)
plt.imshow(clean_img.squeeze(), cmap="gray")
plt.title("Orignal Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(corrupt_img.squeeze(), cmap="gray")
plt.title("Corrupted Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(reconstructed_img, cmap="gray")
plt.title("Reconstructed Image")
plt.axis("off")

plt.show()
