# Virtual Staining Tutorial: KI67 Synthesis with Deep Learning
This tutorial demonstrates how to build and train a deep learning model to virtually stain hematoxylin and eosin (H&E) images into corresponding immunohistochemistry (IHC) stained images using the KI67 marker.

We will walk through:

- Setting up the dataset and preprocessing
- Defining the generator and discriminator models
- Training with a combination of adversarial, pixel-wise, and perceptual losses
- Evaluating and visualizing the results

## Dataset
Data for this tutorial can be accessed from this [Google Drive link](https://drive.google.com/file/d/1ZMJrW1CJ8nRVNVgkQukUSZCGZoajhM6E/view?usp=sharing). It contains:
- Testing images for evaluation
- Samples generated during validation
- Model checkpoints

