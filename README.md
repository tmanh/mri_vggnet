# Computer-Aided Diagnosis of Prostate Cancer Using a Deep Convolutional Neural Network From Multiparametric MRI

This repository is the implementation of Computer-Aided Diagnosis of Prostate Cancer Using a Deep Convolutional Neural Network From Multiparametric MRI, Yang Song et al., 2018.

## Dependencies

```
conda create -n mri python=3.7
conda activate mri
conda install numpy
conda install -c conda-forge nibabel
conda install scikit-image
conda install -c conda-forge scikit-learn
pip install pystackreg
pip install tensorflow-gpu (or tensorflow)
```

## Implementation

The network is implemented with Tensorflow 2.4. To run the code:

```
python main.py
```

## Directory structures

```
.
├── README.md    # this readme
├── engine.py    # implement the train and test function
├── main.py      # run test or train here
├── metrics.py   # implement the metrics
├── models.py    # implement the deep net
└── utils.py     # implement data augmentation
```
