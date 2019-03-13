Overview
-----

The dataset in use is the MNIST dataset, which is divided into three parts to be used accross the different training options:

1. **Labeled trainset** a smaller set of train images, with known target labels. used for semi-supervised training.
2. **Unlabeled trainset** a large set of train images, assumed to be stored without known target labels. used for un-supervised training, or as the un-supervised segment in semi-supervised training.
3. **Validation set** used to validate the model's classification accuracy in the case of semi-supervised training.

**The sizes chosen for the datasets are as follows:**
3000 [labeled trainset]  47000 [un-labeled trainset]  10000 [validation]



Initialize the Datasets
-----

Initializing the datasets requires downloading the MNIST dataset, and its segmentation into the parts described above.
Therefore the dataset initialization can be done only once, using a separate entry point.


      >>> python setup.py install --user
      >>> init_datasets --dir-path <path-to-data-dir>

