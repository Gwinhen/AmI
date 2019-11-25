# Pytorch AmI

The folder contains the Pytorch vecterized implementation of AmI, which is much faster.
Because of the different framework and model weight, the performance is slightly differnt from the original.

```
pip3 list | grep torch
torch (1.2.0)
torchvision (0.4.0)
```

## Vgg Face Model

The model and weight are downloaded from <http://www.robots.ox.ac.uk/~albanie/pytorch-models.html>

## Usage

Basically, after you extract the neurons ([`witness_extraction.ipynb`](/src/pytorch/witness_extraction.ipynb)),
you can use the following code snippet to obtain your AmI model
(details in [`benign_detection.ipynb`](/src/pytorch/benign_detection.ipynb) and 
[`adversary_detection.ipynb`](/src/pytorch/adversary_detection.ipynb)). 

```
from ami_model import AmIModel
device = torch.device('cuda')

model = # YOUR MODEL

attribute_model = AmIModel(model, # AMI Parameters ...)
attribute_model.to(device)
attribute_model.eval()
attribute_model.register_my_hook(ami_data=...)
```
