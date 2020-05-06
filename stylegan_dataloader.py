import torch
import torchvision
from PIL import Image, ImageOps
import os, random, math


class FFHQDataset(torch.utils.data.Dataset):
    """
    A datset which loads images from a flat directory and scales them
    to a size specified when the dataloader is created. We create a new 
    loader every time we change resolutions in training.
    """
    def __init__(self, source_directory, resize_directory, resolution):
        # This will resize the image to the target resolution and normalize to the
        # range [-1, 1] (the Normalize transform will subtract 0.5 from each color
        # channel, then divide by 0.5).
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.directory = os.path.join(resize_directory, str(resolution))

        # Resize the images to each size - an example of how you might do preprocessing
        # and storage on-disk. We still do normalization on the fly, but you could
        # avoid that by storing raw tensors instead of images. The if assumes that if
        # the directory is empty or nonexistent we need to do the resizing, but does
        # not handle partially-full directories or invalid files.
        if not os.path.isdir(self.directory) or len(os.listdir(self.directory)) == 0:
            os.makedirs(self.directory, exist_ok=True)
            originals = [file
                         for file in os.listdir(source_directory)
                         if os.path.isfile(os.path.join(source_directory, file))]
            # We assume all images are 1024 to begin with; handling the general case
            # would really complicate this a lot more than necessary.
            downsample = torchvision.transforms.Resize(resolution,
                                                       interpolation=Image.BILINEAR)
            for sample in originals:
                image = Image.open(os.path.join(source_directory, sample))
                image = downsample(image)
                png_ext = ".".join(sample.split(".")[:-1]) + ".png"
                image.save(os.path.join(self.directory, png_ext), "PNG")

        # Assume that every regular file in the directory is a sample
        self.samples = [
            file
            for file in os.listdir(self.directory)
            if os.path.isfile(os.path.join(self.directory, file))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.directory, self.samples[index]))
        tensor = self.transform(image)
        tensor = torch.flip(tensor, [2]) if random.random() > 0.5 else tensor
        return tensor

class StyleMixer(torch.utils.data.IterableDataset):
    """
    Returns a tensor indicating which style mapping tensor to use in image generation.
    This isn't really a dataset, but the logic is a little too complicated to implement
    cleanly as part of the model for mixing more than 2 styles. This lets us spread the
    work on multiple CPUs without blocking main thread execution and makes this a heck
    of a lot easier to read by sparing us the batch dimension.
    """

    def __init__(self, layer_count, style_count, mixing_prob, mapping_size):
        super().__init__()
        self.layer_count = layer_count
        self.mixing_prob = mixing_prob
        self.mapping_size = mapping_size

        # -1 because there are n-1 boundaries between n partitions; min() in case we
        # specified more transitions than we have AdaIN layers
        self.transition_count = min(layer_count - 1, style_count - 1)

    def __next__(self):
        # We're going to build a tensor that indicates when to transition to each sample
        # from the latent space (swapping between styles). The tensor is n x m x 1,
        # where n is the number of transitions needed and m is the number of AdaIn
        # layers. It is True at (i, j, 0) where the model should switch to style i at
        # layer j, and False everywhere else. The size 1 dimension is to ensure
        # broadcasting happens correctly in the guts of the model.
        transitions_by_style = torch.zeros((self.transition_count, self.layer_count, 1))

        if random.random() > self.mixing_prob:
            # In this case we do no mixing
            return transitions_by_style.bool()

        # Get some randomly-chosen layers after which to swap styles. Specify
        # self.layer_count - 1 to ensure the swap doesn't happen after the last layer,
        # as this would be invisible to the model.
        transition_points, _ = torch.sort(
            torch.randperm(self.layer_count - 1)[:self.transition_count]
        )

        # Set those Trues
        for style in range(self.transition_count):
            transitions_by_style[style][transition_points[style]] += 1

        return transitions_by_style.bool()

    def __iter__(self):
        return self
