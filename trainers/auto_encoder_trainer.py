#
# File:   auto_encoder_trainer.py
# Author: Artem Suprun
# Date:   06/03/2026
# Description: A trainer class for a  generic autoencoder model. 
#

import torch


class AutoEncoderTrainer:
    def __init__(self, model, optimizer, loss_fn, modifier=None):
        # Set device (GPU if available, otherwise CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda") # NVIDIA GPU
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps") # Apple Silicon GPU
        else:
            self.device = torch.device("cpu") # CPU
        
        # Move model to the specified device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.modifier = modifier or []

        # Initialize lists to track training and validation losses
        self.train_losses = []
        self.val_losses = []

    # Train the model for one epoch
    def train_epoch(self, dataloader):
        self.model.train() # Set model to training mode
        total_loss = 0.0

        for batch in dataloader:
            inputs, targets = self._extract_batch(batch, training=True)

            # zero the parameter gradients of the optimizer (makes sure we don't 
            # accumulate gradients from previous batches)
            self.optimizer.zero_grad(set_to_none=True)

            # forward pass            
            outputs = self.model(inputs)

            # compute loss and backpropagate
            loss = self.loss_fn(outputs, targets)
            loss.backward()

            # Update model parameters (model weights)
            self.optimizer.step()

            # Accumulate loss for this batch
            total_loss += loss.item()
        
        # Return the average loss over all batches in the dataloader
        return total_loss / len(dataloader)

    # Evaluate the model on the validation set
    def evaluate(self, dataloader):
        self.model.eval() # Set model to evaluation mode
        total_loss = 0.0

        with torch.no_grad(): # Disable gradient calculation for evaluation
            for batch in dataloader:
                inputs, targets = self._extract_batch(batch, training=False)

                # forward pass
                outputs = self.model(inputs)

                # compute loss
                loss = self.loss_fn(outputs, targets)

                # Accumulate loss for this batch
                total_loss += loss.item()

        # Return the average loss over all batches in the dataloader
        return total_loss / len(dataloader)

    # Main training loop
    # 
    # The fit method trains the model for a specified number of epochs, optionally evaluating 
    # on a validation set after each epoch. It also tracks and prints the training and validation 
    # losses for each epoch.
    #
    # Parameters:
    # - train_loader: A PyTorch DataLoader for the training dataset.
    # - epochs: The number of epochs to train the model.
    # - val_loader: (Optional) A PyTorch DataLoader for the validation dataset. If provided, the 
    # model will be evaluated on this dataset after each epoch.
    #
    # Note: if you want to change the batch size, you should do it in the DataLoader, not here. 
    # The fit method will work with any batch size as long as the DataLoader is set up correctly. 
    def fit(self, train_loader, epochs, val_loader=None):
        # Loop over the specified number of epochs, training and evaluating the model each epoch
        for epoch in range(epochs):
            # Train the model for one epoch and store the training loss for this epoch
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # If a validation dataloader is provided, evaluate the model and store 
            # the validation loss for this epoch
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self.val_losses.append(val_loss)

                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")

        return {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
        }

    # Save the trained model to a file
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        return None
    
    # Load a trained model from a file
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        return None
    
    def _extract_batch(self, batch, training=True):
        # Extract inputs and targets from the batch
        if isinstance(batch, (list, tuple)):
            inputs = batch[0]
        else:
            inputs = batch
        
        inputs = inputs.to(self.device)
        targets = inputs.clone()

        # apply the modifier to the inputs if it is defined (e.g., for data augmentation)
        if training and self.modifier:
            inputs = self.apply_modifier(inputs)

        return inputs, targets
    
    # Apply the modifier functions to the inputs. This can be used for data 
    # augmentation or corruption during training.
    #
    # Note: the modifier functions should be defined to take a batch of inputs and 
    # return the modified batch.
    def apply_modifier(self, inputs):
        modified_inputs = inputs
        for modifier in self.modifier:
            modified_inputs = modifier(modified_inputs)

        return modified_inputs

# Example modifier functions that can be used to corrupt the input images during training.
def gaussian_noise_modifier(inputs, mean=0.0, std=0.1):
    noise = torch.randn_like(inputs) * std + mean
    return inputs + noise

def random_mask_modifier(inputs, mask_prob=0.1):
    mask = torch.rand_like(inputs) > mask_prob
    return inputs * mask

# 
# Example usage:
#trainer = AutoEncoderTrainer(
#    model,
#    optimizer,
#    loss_fn,
#    modifiers=[
#        lambda x: gaussian_noise_modifier(x, 0.15, 0.05),
#        lambda x: random_mask_modifier(x, 0.25)
#    ]
#)
#
