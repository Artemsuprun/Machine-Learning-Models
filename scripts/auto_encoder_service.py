import random
import matplotlib.pyplot as plt
from torchvision import transforms

from datasets.general import get_datasets
from services.auto_encoder_service import AutoEncoderService


# Create the inference service
service = AutoEncoderService(
    checkpoint_path="./checkpoints/autoencoder_mnist.pth",
    encoder_dim=[784, 128, 64],
    activation="relu",
    output_activation="sigmoid",
)

# Dataset transforms
clean_transform = transforms.Compose([
    transforms.ToTensor(),
])

corrupt_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.7, 2.0)),
])

# Load datasets
clean_test_set = get_datasets("mnist", clean_transform)[1]
corrupt_test_set = get_datasets("mnist", corrupt_transform)[1]

# Pick a random sample
idx = random.randint(0, len(clean_test_set) - 1)

clean_img, _ = clean_test_set[idx]
corrupt_img, _ = corrupt_test_set[idx]

# Run inference
reconstructed = service.reconstruct(corrupt_img).squeeze()

# Plot
fig, axes = plt.subplots(1, 3, figsize=(8, 4))

titles = ["Original", "Corrupted", "Reconstructed"]
images = [clean_img, corrupt_img, reconstructed]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
