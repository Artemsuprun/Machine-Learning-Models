from models.auto_encoder import AutoEncoder
from trainers.auto_encoder_trainer import AutoEncoderTrainer
from datasets.input_target_dataset import InputTargetDataset
from datasets.general import get_datasets

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# set up data loaders for MNIST dataset
train_dataset, test_dataset = get_datasets("mnist")

blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.7, 2.0))
train_dataset = InputTargetDataset(
    base_dataset=train_dataset,
    corruptions=[blur]
)
test_dataset = InputTargetDataset(
    base_dataset=test_dataset,
    corruptions=[blur]
)

train_loader = DataLoader(
    train_dataset,
    batch_size= 128,
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

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
    loss_fn
)

# train the model
trainer.fit(train_loader, 20, test_loader)

# save the trained model to the specified path
trainer.save_model("./checkpoints/autoencoder_mnist.pth")

