# Similarity Based Label Smoothing For Dialogue Generation
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the implementation of the paper:

## [**Similarity Based Label Smoothing For Dialogue Generation**](https://lcs2.in/ICON-2022/conference.html)
[**Sougata Saha**](https://www.linkedin.com/in/sougata-saha-8964149a/), [**Souvik Das**](https://www.linkedin.com/in/souvikdas23/), [**Rohini Srihari**](https://www.acsu.buffalo.edu/~rohini/) 

The 19th International Conference on Natural Language Processing (ICON 2022: IIIT Delhi, India)

## Abstract
Generative neural conversational systems are typically trained by minimizing the entropy loss between the training "hard" targets and the predicted logits. Performance gains and improved generalization are often achieved by employing regularization techniques like label smoothing, which converts the training "hard" targets to soft targets. However, label smoothing enforces a data-independent uniform distribution on the incorrect training targets, leading to a false assumption of equiprobability. In this paper, we propose and experiment with incorporating data-dependent word similarity-based weighing methods to transform the uniform distribution of the incorrect target probabilities in label smoothing to a more realistic distribution based on semantics. We introduce hyperparameters to control the incorrect target distribution and report significant performance gains over networks trained using standard label smoothing-based loss on two standard open-domain dialogue corpora.

## Training and Inference Details
Steps to recreate the results:
1. Download the training, validation and testing data along with the experiment parameters csv file. The data can be found in this link: https://drive.google.com/drive/folders/18VKDa4cB8gW8pMypARc6pyBWGFFUIPKw?usp=sharing (Note: you can use gdown to download the datasets. The folder size is approx 11 GB!)
2. Run the similarity_based_label_smoothing_training_inference_notebook.ipynb notebook, which contains the training, validation and inference code (Set the PATH variable appropriately). 

Notes:
1. The model and utilities to load the models can be found in Models.py and utils.py respectively.
2. The trained models from all the experiments along with the results can be found in the models and results folders respectively, in the previously shared link.
