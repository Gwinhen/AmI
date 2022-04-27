# Attacks Meet Interpretability

This repository is for NeurIPS 2018 spotlight paper [Attacks Meet Interpretability: Attribute-steered Detection of Adversarial Samples](http://papers.nips.cc/paper/7998-attacks-meet-interpretability-attribute-steered-detection-of-adversarial-samples.pdf).

**Update:** for Pytorch implementation, please refer to [`src/pytorch`](/src/pytorch).

## Prerequisite

* [opencv-python](https://pypi.org/project/opencv-python/)
* [dlib](https://pypi.org/project/dlib/)
* [caffe](http://caffe.berkeleyvision.org/)

## Setup

* Please download VGG-Face caffe model from [here](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/).
* Unzip the model under `data/` folder.

## Usage

### Attribute Mutation

In `attribute_mutation.ipynb`, attribute-substituted and attribute-preserved images are produced for the base image. Four attributes are encoded with indices from 0 to 3. See the following table for details. Please use `attributes[index]` for corresponding attributes.

| Attribute | Index |
|:---------:|:-----:|
| left eye  |   0   |
| right eye |   1   |
| nose      |   2   |
| mouth     |   3   |

Two actions are also encoded with indices, which is listed in the following table. Please use `actions[index]` for corresponding actions.

|    Action    | Index |
|:------------:|:-----:|
| substitution |   0   |
| preservation |   1   |

Generated images are saved in folder `data/attribute_mutated/[attribute]_[action]/`.

### Attribute Witness Extraction

Attribute witnesses are extracted layer by layer based on attribute-substituted and attribute-preserved images. Please find the implementation in `witness_extraction.ipynb`. Extracted witnesses are saved in folder `data/witnesses/`.

### Attribute-steered Model

With extracted attribute witnesses, neuron weakening and strengthening are applied for each input during execution. Adversary detection is achieved by observing the final prediction from attribute-steered model comparing to the original model. Detailed implementation is in `adversary_detection.ipynb`.

7 adversarial attacks are included in folder `data/attacks`. Please change `attack_path` in the code to test on different attacks.

## Citation

Please cite for any purpose of usage.

    @inproceedings{NeurIPS2018_7998,
        title={Attacks Meet Interpretability: Attribute-steered Detection of Adversarial Samples},
        author={Tao, Guanhong and Ma, Shiqing and Liu, Yingqi and Zhang, Xiangyu},
        booktitle={Advances in Neural Information Processing Systems 31},
        pages = {7728--7739},
        year={2018}
    }
