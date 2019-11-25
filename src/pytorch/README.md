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

## Note

You should tune the AmI parameters on your own dataset to get a predefined false positive rate (such as 9.91%) before deploying it to detect adversary.

## Performance

| Data                          | Original AmI (%)  | Pytorch AmI (%)  |
| ----------------------------- | ----------------- | ---------------- |
| Benign (False Positive Rate)  | 9.91              | 9.62             |
| Patch first (Detection Rate)  | 97                | 83               |
| Patch next (Detection Rate)   | 98                | 94               |
| Glasses first (Detection Rate)| 85                | 83               |
| Glasses next (Detection Rate) | 85                | 81               |
| CW l0 first (Detection Rate)  | 91                | 99               |
| CW l0 next (Detection Rate)   | 95                | 98               |
| CW l2 first  (Detection Rate) | 99                | 100              |
| CW l2 next (Detection Rate)   | 99                | 98               |
| CW li first (Detection Rate)  | 97                | 96               |
| CW li next (Detection Rate)   | 100               | 100              |
| FGSM (Detection Rate)         | 91                | 89               |
| BIM (Detection Rate)          | 90                | 87               |
