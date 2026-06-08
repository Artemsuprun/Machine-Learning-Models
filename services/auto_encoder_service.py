#
# File:   auto_encoder_service.py
# Author: Artem Suprun
# Date:   06/05/2026
# Description: An interface class to handle the inference of the 
#  auto_encoder model.
#


class AutoEncoderService:
    def __init__(self, model):
        # device setup
        self.device 

        # model for making inferences for
        self.model = model
    
    def reconstruct(self, img):
        pass

    def encode(self, image):
        pass

    def decode(self, bottle_neck):
        pass