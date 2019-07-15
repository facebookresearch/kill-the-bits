# And the bit goes down

This repository contains the implementation of our paper: [And the bit goes down: Revisiting the quantization of neural networks](https://arxiv.org/abs/1907.05686) as well as the compressed models we obtain (ResNets and Mask R-CNN).

Our compression method is based on vector quantization. It takes as input an already trained neural network and, through a distillation procedure at all layers and a fine-tuning stage, optimizes the accuracy of the network.

This  approach  outperforms  the  state-of-the-art  w.r.t.  compression/accuracy  trade-off for standard networks like ResNet-18 and ResNet-50 (see  [Compressed models](#Compressed-Models)).

<p align="center">
<img src="illustration.png" alt="Illustration of our method. We approximate a binary classifier $\varphi$ that labels images $x$ as dogs or cats by quantizing its weights.
    \textbf{Standard method}: quantizing $\varphi$ with the standard objective function \eqref{eq:pq_obj} leads to a classifier $\widehat \varphi_{\text{bad}}$ that tries to approximate $\varphi$ over the entire input space and thus performs badly for in-domains inputs.
    \textbf{Our method}: quantizing $\varphi$ with our objective function \eqref{eq:ours_obj} leads to a classifier  $\widehat \varphi_{\text{good}}$ that performs well for in-domain inputs.
    \benjamin{Maybe move the $\widehat \varphi_{\text{bad}}$ and $\widehat \varphi_{\text{good}}$ labels from the top left corner into the image close to the respective lines. Then the $\phi(x)$ can come down a little."">
</p>


## Installation

Our code works with Python 3.6 and newest. To run the code, you must have the following packages installed:
- [NumPy](http://www.numpy.org/)
- [PyTorch](http://pytorch.org/) (version=1.0.1.post2)

These dependencies can be installed with:
`
pip install -r requirements.txt
`

## Compressed Models
The compressed models (centroids + assignments) are available in the `models/compressed` folder. We provide code to evaluate those models on their standard benchmarks (ImageNet/COCO). Note that inference can be performed both on GPU or on CPU. Note also that we did not optimize this precise part of the code for speed. Indeed, the code for inference should rather be regarded as a proof of concept: based on the centroids and the assignments, we recover the accuracies mentioned in the table above by instantiating the full, non-compressed model.

### Vanilla ResNets
We provide the vanilla compressed ResNet-18 and ResNet-50 models for 256 centroids in the low and high compression regimes. As mentioned in the paper, the low compression regime corresponds to a block size of 9 for standard 3x3 convolutions and to a block size of 4 for 1x1 pointwise convolutions. Similarly, the high compression regime corresponds to a block size of 18 for standard 3x3 convolutions and to a block size of 8 for 1x1 pointwise convolutions.

|Model (non-compressed top-1) | Compression | Size ratio | Model size | Top-1 (%)|   
|:-:|:-:|:-:|:-:|:--:|
ResNet-18 (69.76%) | Small blocks <br>Large blocks | 29x <br>43x |1.54MB<br>1.03MB|**65.81**<br>**61.18**
ResNet-50 (76.15%) | Small blocks <br>Large blocks | 19x <br>31x |5.09MB<br>3.19MB|**73.79**<br>**68.21**


To evaluate on the standard test set of ImageNet: clone the repo, `cd` into `src/` and run:
```bash
python inference.py --model resnet18 --state-dict-compressed models/compressed/resnet18_small_blocks.pth --device cuda --data-path YOUR_IMAGENET_PATH
```

### Semi-supervised ResNet50
We provide the compressed [semi-supervised ResNet50](https://arxiv.org/abs/1905.00546) trained and open-sourced by Yalniz *et. al.* We use 256 centroids and the small blocks compression regime.


|Model (non-compressed top-1) | Compression | Size ratio | Model size | Top-1 (%)|   
|:-:|:-:|:-:|:-:|:--:|
Semi-Supervised ResNet-50 (76.15%) | Small blocks| 19x | 5.15MB | **76.12**

To evaluate on the standard test set of ImageNet: clone the repo, `cd` into `src/` and run:
```bash
python inference.py --model resnet50_semisup --state-dict-compressed models/compressed/resnet50_semisup_small_blocks.pth --device cuda --data-path YOUR_IMAGENET_PATH
```

### Mask R-CNN

We provide the compressed Mask R-CNN (backbone ResNet50-FPN) available in the [PyTorch Model Zoo](https://pytorch.org/docs/stable/torchvision/models.html). As mentioned in the paper, we use 256 centroids and various block sizes to reach an interesting size/accuracy tradeoff (with a 26x compression factor). Note that you need [torchvision 0.3](https://pytorch.org/blog/torchvision03/) in order to run this part of the code.

|Model | Size | Box AP| Mask AP |   
|:-:|:-:|:-:|:-:|
|Non-compressed | 170MB | 37.9 | 34.6|
|Compressed | 6.51MB | 33.9 | 30.8 |

To evaluate on COCO: clone the repo, run `git checkout mask_r_cnn`, `cd` into `src/` and run:
```bash
python inference.py --model maskrcnn_resnet50_fpn --state-dict-compressed models/compressed/mask_r_cnn.pth --device cuda --data-path YOUR_COCO_PATH
```

## Results

You can also compress the vanilla ResNet models and reproduce the results of our paper by `cd` into `src/` and by running the following commands:
- For the *small blocks* compression regime:
```bash
python quantize.py --model resnet18 --block-size-cv 9 --block-size-pw 4 --n-centroids-cv 256 --n-centroids-pw 256 --n-centroids-fc 2048 --data-path YOUR_IMAGENET_PATH
python quantize.py --model resnet50 --block-size-cv 9 --block-size-pw 4 --n-centroids-cv 256 --n-centroids-pw 256 --n-centroids-fc 1024 --data-path YOUR_IMAGENET_PATH
```
- For the *large blocks* compression regime:
```bash
python quantize.py --model resnet18 --block-size-cv 18 --block-size-pw 4 --n-centroids-cv 256 --n-centroids-pw 256 --n-centroids-fc 2048 --data-path YOUR_IMAGENET_PATH
python quantize.py --model resnet50 --block-size-cv 18 --block-size-pw 8 --n-centroids-cv 256 --n-centroids-pw 256 --n-centroids-fc 1024 --data-path YOUR_IMAGENET_PATH
```
Note that the vanilla ResNet-18 and ResNet-50 teacher (non-compressed) models are taken from the PyTorch model zoo. Note also that we run our code on a single 16GB Volta V100 GPU.

## License
This repository is released under Creative Commons Attribution 4.0 International (CC BY 4.0) license, as found in the LICENSE file.

## Bibliography
Please consider citing [1] if you found the resources in this repository useful.

[1] Stock, Pierre and Joulin, Armand and Gribonval, Rémi and Graham, Benjamin and Jégou, Hervé. [And the bit goes down: Revisiting the quantization of neural networks](https://arxiv.org/abs/1907.05686).
```
@article{
  title = {And the bit goes down: Revisiting the quantization of neural networks},
  author = {Stock, Pierre and Joulin, Armand and Gribonval, R{\'e}mi and Graham, Benjamin and J{\'e}gou, Herv{\'e}}
  journal={arXiv e-prints},
  year = {2019}
}
```
