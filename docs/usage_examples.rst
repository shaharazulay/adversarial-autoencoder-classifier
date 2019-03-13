Initializing the Datasets
-----

      >>> python setup.py install --user
      >>> init_datasets --dir-path <path-to-data-dir>

As a result the three datasets will be created in three different files inside the data directory:
*train_labeled.p, train_unlabeled.p, validation.p*

Semi-supervised Training
-----

      >>> python setup.py install --user
      >>> train_semi_supervised --dir-path <path-to-data-dir> --n-epochs 35 --z-size 2 --n-classes 10 --batch-size 100
      
As a result the following files will be created inside the data directory:

1. **Decoder / Encoder networks** the Encoder-Decoder networks (Q, P) will be stored for future use and analysis under *decoder_semi_supervised* and *encoder_semi_supervised*
2. **Learning curves** as *png* images, describing the adversarial, reconstruction and semi-supervised classification learning curves.
      

Un-supervised Training
-----

      >>> python setup.py install --user
      >>> train_unsupervised --dir-path <path-to-data-dir> --n-epochs 35 --z-size 2 --n-classes 10 --batch-size 100
      
As a result the following files will be created inside the data directory:

1. **Decoder / Encoder networks** the Encoder-Decoder networks (Q, P) will be stored for future use and analysis under *decoder_unsupervised* and *encoder_unsupervised*
2. **Mode Decoder network** responsible for the reconstruction of the image based on the categorial latent y, will be stored under *mode_decoder_unsupervised*
3. **Learning curves** as *png* images, describing the adversarial, reconstruction and mutual information learning curves.
      

Visualization of the Learned Model
-----
