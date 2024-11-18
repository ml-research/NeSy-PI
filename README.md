# NeSy-PI

---
This repository is the official implementation of the paper "Neuro-Symbolic Predicate Invention: Learning Relational Concepts from Visual Scenes"

---



## Setup Locally

1. Install Python 3.8
2. Install Pytorch 1.13.1
3. `pip install -r requirements.txt`

---

## Setup Remotely 
To run the experiments remotely, we provide a docker file `Dockerfile` in the root folder.
###### Build docker

``` 
docker build -t nesy_pi_docker .
```

###### Run docker
Using flag `-v` to save the dataset folder `storage` to a perferable place.

``` 
docker run --gpus all -it -v path/to/storage:/nesypi/storage --rm nesy_pi_docker
```

---

## Dataset
The dataset is saved [here](https://huggingface.co/datasets/akweury/NeSy-PI).
Please download the dataset and unzip the folder to `NeSy-PI-main\storage`.

---

## Experiments

#### Run Experiments
Check [Readme](src/README.md) file in folder `NeSy-PI/src`.


## Citation
If you have interest about this work, please consider cite:

```
@article{nesy_pi,
	author = {Sha, Jingyuan and Shindo, Hikaru and Kersting, Kristian and Dhami, Devendra Singh},
	journal = {Neurosymbolic Artificial Intelligence},
	number = {Preprint},
	pages = {1--26},
	title = {Neuro-symbolic Predicate Invention: Learning relational concepts from visual scenes},
	volume = {Preprint},
	year = {2024}}
```
