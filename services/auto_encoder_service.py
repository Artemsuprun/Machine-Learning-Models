#
# File:   auto_encoder_service.py
# Author: Artem Suprun
# Date:   06/05/2026
# Description: An interface class to handle the inference of the 
#  auto_encoder model.
#

import torch
from utils.device import get_device


class AutoEncoderService:
    def __init__(self, model):
        self.device = torch.device(get_device()) # get the device type
        self.model = model.to(self.device) # move the given model to the device
        self.model.eval()

    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    @torch.inference_mode()
    def predict(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)

        return self.model(x)
    
    @torch.inference_mode()
    def encode(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)

        return self.model.encode(x)
    
    @torch.inference_mode()
    def decode(self, latent):
        latent = latent.to(self.device)
        return self.model.decode(latent)
    
    @torch.inference_mode()
    def reconstruct(self, x, return_latent=False):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)

        latent = self.model.encode(x)
        recon = self.model.decode(latent)

        recon = recon.view_as(x)

        if return_latent:
            return recon, latent
        
        return recon
    
    





    def __init__(self, model, device=None, eval_mode=True):
        self.device = torch.device(device if device is not None else get_device())
        self.model = model.to(self.device)

        if eval_mode:
            self.model.eval()

    def load_weights(self, path, map_location=None):
        map_location = map_location or self.device
        self.model.load_state_dict(torch.load(path, map_location=map_location))
        self.model.to(self.device)
        return self

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)
        return self

    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def reconstruct(self, img):
        """Reconstruct image(s) using the autoencoder."""
        image = self._prepare_input(img)
        with torch.no_grad():
            reconstructed = self.model(image)
        return reconstructed.detach().cpu()

    def encode(self, image):
        """Encode image(s) into the latent bottleneck representation."""
        image = self._prepare_input(image)
        with torch.no_grad():
            latent = self.model.encode(image)
        return latent.detach().cpu()

    def decode(self, bottle_neck):
        """Decode latent bottleneck representation back into image space."""
        if not isinstance(bottle_neck, torch.Tensor):
            bottle_neck = torch.as_tensor(bottle_neck, dtype=torch.float32)
        bottle_neck = bottle_neck.to(self.device)

        if bottle_neck.dim() == 1:
            bottle_neck = bottle_neck.unsqueeze(0)

        with torch.no_grad():
            decoded = self.model.decode(bottle_neck)
        return decoded.detach().cpu()

    def _prepare_input(self, image):
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image, dtype=torch.float32)

        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)

        return image.to(self.device)
