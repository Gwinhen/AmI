# Attacks Meet Interpretability

## Setup

* Please download VGG-Face caffe model from http://www.robots.ox.ac.uk/~vgg/software/vgg_face/.
* Unzip the model under `data/` folder.

## Usage

### `attribute_mutation.ipynb`

Code `attribute_mutation.ipynb` produces attribute-substituted and attribute-preserved images for `base_img.jpg`. Four attributes (left eye, right eye, nose, mouth) are encoded in order with corresponding indices from 0 to 3. Two actions (substitute, preserve) are also encoded in order in the code. Generated images are saved under `data/attribute_mutated/[attribute]_[action]/` folder.

### `witness_extraction.ipynb`

Code `witness_extraction.ipynb` extracts attribute witnesses for each attribute layer by layer. Extracted witnesses are saved under `data/witnesses/` folder.

### `adversary_detection.ipynb`

Adversarial samples are under `data/attacks` folder. Please change `attack_path` in the code to test on different attacks.