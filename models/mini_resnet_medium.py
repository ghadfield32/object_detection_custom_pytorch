# https://python.plainenglish.io/single-object-detection-with-pytorch-step-by-step-96430358ae9d

#%%writefile mini_resnet_medium.py
# ResNet model
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.nn as nn
from torchinfo import summary

class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.base1 = nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding='same'),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(True) 
    )
    self.base2 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )

  def forward(self, x):
    x = self.base1(x) + x
    x = self.base2(x)
    return x
  
class ResNet(nn.Module):
  def __init__(self, in_channels, first_output_channels):
    super().__init__()
    self.model = nn.Sequential(
        ResBlock(in_channels, first_output_channels),
        nn.MaxPool2d(2),
        ResBlock(first_output_channels, 2 * first_output_channels),
        nn.MaxPool2d(2),
        ResBlock(2 * first_output_channels, 4 * first_output_channels),
        nn.MaxPool2d(2),
        ResBlock(4 * first_output_channels, 8 * first_output_channels),
        nn.MaxPool2d(2),
        nn.Conv2d(8 * first_output_channels, 16 * first_output_channels, kernel_size=3),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(7 * 7 * 16 * first_output_channels, 2)
    )
  
  def forward(self, x):
    return self.model(x)

def create_resnet_model(in_channels, first_output_channels, input_size=(256, 256), print_summary=False):
    """
    Creates a ResNet model and optionally prints its summary.

    Args:
        in_channels (int): Number of input channels.
        first_output_channels (int): Number of output channels for the first layer.
        input_size (tuple): The height and width of the input image.
        print_summary (bool): If True, prints the model summary.
        
    Returns:
        nn.Module: The created ResNet model.
    """
    try:
        model = ResNet(in_channels, first_output_channels)
        if print_summary:
            summary(model=model, 
                    input_size=(8, in_channels, *input_size), 
                    col_names=["input_size", "output_size", "num_params"],
                    col_width=20,
                    row_settings=["var_names"])
        return model
    except Exception as e:
        print(f"An error occurred while creating the model: {e}")
        return None

#Usage in a another file
#from resnet_model import create_resnet_model

# Create the model with summary
#model_with_summary = create_resnet_model(in_channels=3, first_output_channels=16, print_summary=True)

# Create the model without summary
#model_without_summary = create_resnet_model(in_channels=3, first_output_channels=16, print_summary=False)

