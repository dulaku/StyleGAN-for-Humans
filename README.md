## StyleGAN For Humans
A Pytorch implementation of the model presented in "A Style-Based Generator Architecture for Generative Adversarial Networks". No training videos this time - I've run this enough to be pretty sure it's working, but I can't justify pulling my machine off COVID19 folding long enough to get a convincing video to upload.

This is a sequel to [ProGAN For Humans](https://github.com/dulaku/ProGAN-for-Humans). The goal here is for the code to be easily interpretable, with a lot of comments explaining why we're doing things, under the assumption the reader isn't already an expert. A large amount of code here is directly reused; the major updates have been made to:</br>

* `train.py`
  * Extra logic has been added to support style mixing, under the `if pretrained:` condition and anywhere you see `style_mixer` and `style_mixes` - we use a new data loader to generate tensors that tell the model when to swap between mapping network outputs for style mixing.
  * `optimizer_G` has had a separate learning rate set for the mapping network.
  * Images are resized with bilinear interpolation during the fade-in phase of a new block.
* `stylegan_models.py`
  * The generator is almost totally rewritten to support the new architecture.
  * The discriminator's `forward` method has been revised to support the new loss - as a side effect, the old `gradient_penalty` method is gone.
* `stylegan_layers.py`
  * The convolution and transpose convolution layers now have an extra trick to incorporate a Gaussian blur at the same time as rescaling - compared to doing this the correct way, this introduces a small error in a 1-pixel border of the featuremap, but it's correct everywhere else.
  * There is an `EqualizedLinear` layer to support the mapping network, and new `AdaIN` and `NoiseInput` layers.
* `stylegan_dataloader.py`
  * There is a new `StyleMixer` dataloader, which produces the tensors mentioned in `train.py`'s extra logic.

The recommended reading order is `train.py`, `stylegan_dataloader.py`, `stylegan_models.py`, `stylegan_layers.py`. You'll want to understand what the style mixer is doing for the style mixing logic in the model itself to make sense. This doesn't feel particularly intuitive or readable, so suggestions welcome.</br>

Compared to the code in the original project, there are 4 major differences:</br>

* I have not implemented the truncation trick for visualizing images, but I have pointed out where it would need to go if you wanted to use it. The trick loses image diversity but keeps the model from generating images on outlier inputs, so you're at a slightly higher risk of getting bizarre outputs.
* The Gaussian blur trick was not part of the original code base and is technically incorrect, but it was necessary in order to keep the model size down. In practice I suspect this won't cause much trouble, but in case it's a problem for you, I've left in a layer that applies the blur separately and instructions for using it.
* The original paper discusses mixing up to 4 styles, but its code only ever uses 2. I went ahead and implemented mixing of arbitrarily many styles (up to the number of AdaIN layers in the model).
* The code does not implement the additional training after reaching full resolution that is used in the paper - to do this yourself, train for an extra 375 epochs (on a 30000 image dataset).

As before, readability of the code was the highest priority, so some features are lacking and I would not expect a production-ready code base. In particular:</br>

* The code is not platform agnostic - it assumes a machine with 16 CPU cores and 2 GPUs with 8 GB VRAM apiece. Instead, comments point out these assumptions when they are made and gives guidance on changing them for different machines.
* The code assumes that the dataset consists of 1024x1024 images and does not check to ensure that there is adequate disk space for preprocessed images.
* Optimizations were generally only performed when it teaches a common PyTorch idiom or helps fit the model onto the machine.

In general, I try to assume you have the paper on hand - comments don't generally explain "why"s that are covered in the paper, but instead try to give enough information that you know what part of the paper is relevant. Likewise, the code assumes a basic level of familiarity with convolutional neural networks. If you don't have that yet, I strongly recommend the following resources:</br>

* For an introduction to deep learning in general:
  * [MIT Course 6.034 Lecture 12A: Neural Nets](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/lecture-videos/lecture-12a-neural-nets)
  * [MIT Course 6.034 Lecture 12B: Deep Neural Nets](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/lecture-videos/lecture-12b-deep-neural-nets)

* For an introduction to convolutional neural networks and modern training techniques:
  * [Stanford CS234n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)

# Requirements:

The following Python modules are required. I recommend installing the Python requirement with ``venv`` or something similar.</br>
* `Pillow`
* `numpy`
* `torch`
* `torchvision`

You will also need a dataset of 1024x1024 images. In principle, you should use the [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset). In practice, I've been trying to download this thing for a solid month but Google Drive has a quota that has kept me from succeeding.</br>

You may also find it helpful to set the environment variable `CUDA_VISIBLE_DEVICES=1,0` - your OS may be using some VRAM on your 0th GPU, so reversing them for Pytorch can give you just a bit more room on the GPU used for unparallelized scratch space.</br>
