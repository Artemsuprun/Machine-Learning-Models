from models.auto_encoder import AutoEncoder
from trainers.auto_encoder_trainer import AutoEncoderTrainer
from datasets.general import get_loaders

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# set up data loaders for MNIST dataset
train_loader, test_loader = get_loaders("mnist")

# create the model
model = AutoEncoder(
    encoder_dim=[784, 128, 64], # input dimension is 28*28=784 for MNIST
    activation="relu", 
    output_activation="sigmoid"
)

# create the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# create the trainer and specify the modifier
trainer = AutoEncoderTrainer(
    model,
    optimizer,
    loss_fn,
    modifier=[
        lambda x: transforms.GaussianBlur(kernel_size=7, sigma=(0.25, 2.0))(x), # add blurring
    ]
)

# train the model
trainer.fit(train_loader, 20, test_loader)

# save the trained model to the specified path
trainer.save_model("./checkpoints/autoencoder_mnist.pth")

