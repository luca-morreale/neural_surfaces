# Neural Convolutional Surfaces (NCS)

[Paper](https://arxiv.org/pdf/2204.02289.pdf)
&nbsp;&nbsp;
[Project Page](https://geometry.cs.ucl.ac.uk/projects/2022/cnnmaps/)

---

This repository contains code to train a Neural Convolutional Surface and a [Neural Surface Maps](https://geometry.cs.ucl.ac.uk/projects/2021/neuralmaps/).

In the future it will be expanded to improve efficiency, include surface-to-surface maps training and more! Keep an eye out!


## Pre-processing
1. download a mesh (.obj or .ply file)
2. extract the sample out of it

Extract a sample for NCS:

``` python -m scripts.process_patch_sample --data /path/to/mesh --patch_size 0.04 --global_param```

Extract a sample for NCS w/o global parametrization (no coarse):

``` python -m scripts.process_patch_sample --data /path/to/mesh --patch_size 0.04```

Extract a sample for [Neural Surface Maps](https://geometry.cs.ucl.ac.uk/projects/2021/neuralmaps/):

``` python -m scripts.process_surface_sample --data /path/to/mesh --square```

## Surface Representation

Training a Neural Convolutional Surface is just one line of code:

``` python -m mains.training experiment/overfit/experiment.json ```

The ```experiment.json``` file must contain all the information related to the experiment to run, such as:
- path for the sample file
- type of dataset to use
- what model to train (w/ hyper-parameters)
- what kind of task to run (i.e. what kind of training)
- what kind of training loop (related to the task in general)
- where to save the tensorboard logs
- where to save the meshes as output

Fret not, the experiment folder contains examples for everything you need!
For example ```experiment/overfit/bimba_nsm.json``` trains a [Neural Surface Maps](https://geometry.cs.ucl.ac.uk/projects/2021/neuralmaps/), while ```experiment/overfit/bimba_ncs.json``` trains a Neural Convolutional Surface.
Please, remember to change the sample paths and for the checkpoints in the experiment file.

Play around with the hyper-parameters to find what best suits your application.


## Models

In this repository there is an implementation for both Neural Convolutional Surfaces and [Neural Surface Maps](https://geometry.cs.ucl.ac.uk/projects/2021/neuralmaps/).
Different models (with tunable hyper-parameters) are available in the ```models``` folder.

- ```NeuralResConvSurface``` is the model for NCS
- ```ResidualMLP``` is the model for NSM



## Tasks
Each task corresponds to a different type of training, e.g., NCS, NSM or surface maps.
For example, ```ConvSurfaceTrainer``` trains a Neural Convolutional Surface.

For each training task there is a corresponding checkpointing task that saves the surface as mesh, as binary file and can even render it at the end of training.
Please note, all these operations require libraries. If these libraries are not present a warning will be printed on screen but the process won't halt.


## Chamfer

To compute the chamfer distance between two meshes run the following:

```python -m scripts.compute_chamfer --gt /path/to/gt/mesh --mesh /path/to/output/mesh```


## Bibtex

```
@inproceedings{morreale2022neural,
  title={Neural Convolutional Surfaces},
  author={Morreale, Luca and Aigerman, Noam and Guerrero, Paul and Kim, Vladimir G. and Mitra, Niloy J.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
