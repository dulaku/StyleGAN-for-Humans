import torch
import itertools

import stylegan_layers
from config import cfg as config

class AccessibleDataParallel(torch.nn.DataParallel):
    """
    A slight modification of PyTorch's default DataParallel wrapper object; this allows
    us to access attributes of a DataParallel-wrapped module.
    """
    def __getattr__(self, name):
        try:
            # Return an attribute defined in DataParallel
            return super().__getattr__(name)
        except:
            # Otherwise return the wrapped module's attribute of the same name
            return getattr(self.module, name)

class Generator(torch.nn.Module):
    """
    An image generator as described in the StyleGAN paper. This model is composed of a
    set of blocks, each of which are trained in sequence. The first block converts a 1D
    input vector into a 4x4 featuremap; all other blocks upscale by a factor of 2 and
    apply additional convolution layers. Each block uses leaky ReLU activation (0.2 * x
    for x < 0, x otherwise) and pixelwise normalization (see the Pixnorm layer).

    Each block also has a toRGB layer which converts the output of that block to
    the RGB color space.
    """

    def __init__(self):
        super().__init__()

        # This keeps track of which layers within each block are AdaIN layers, since
        # AdaIn layers have a different calling signature and need style mixing
        # applied after. We can't track the objects themselves since the objects are
        # different after moving to GPU.
        self.adaIN_layers = [(2, 6)]

        # The design we use has a separate object for each block, instead of stringing
        # all the layers together into a single sequential model. The main drawback is
        # that we have to convert between "what is this layer's index in its block",
        # which is what the model knows, and "which AdaIN layer is this in the overall
        # network", which is how we specify our style mixing transitions. We can fix
        # this with a redesign of the format for specifying style mixing points, and
        # probably will for StyleGAN2. In the short term, we keep track of offsets
        # manually, like a chump.
        self.adaIN_offsets = [0]

        self.toRGBs = []
        self.blocks = []

        # A learned 512x4x4 (CxHxW) tensor that is modulated to produce results - it
        # does not depend on the latent sample. Variable name because it presumably
        # encodes features common to all generated images, some kind of Ur-Face. Starts
        # as a tensor of 1s, as in the StyleGAN source.
        self.ur_face = torch.nn.Parameter(
            torch.ones((config.latent_dim, 4, 4)), requires_grad=True
        )

        # An 8 layer feed-forward network. Any deeper and this'd be a loop.
        # Perhaps it ought to be anyway, as the middle 6 are all the same.
        self.mapping_network = torch.nn.Sequential(
            # The pixnorm layer converts gaussian noise inputs to "points on a
            # 512-dimensional hypersphere" as noted in the paper. Where 512 is the
            # latent space size.
            stylegan_layers.Pixnorm(),
            stylegan_layers.EqualizedLinear(config.latent_dim, config.mapping_layer_size),
            torch.nn.LeakyReLU(0.2, inplace=True),
            stylegan_layers.EqualizedLinear(config.mapping_layer_size, config.mapping_layer_size),
            torch.nn.LeakyReLU(0.2, inplace=True),
            stylegan_layers.EqualizedLinear(config.mapping_layer_size, config.mapping_layer_size),
            torch.nn.LeakyReLU(0.2, inplace=True),
            stylegan_layers.EqualizedLinear(config.mapping_layer_size, config.mapping_layer_size),
            torch.nn.LeakyReLU(0.2, inplace=True),
            stylegan_layers.EqualizedLinear(config.mapping_layer_size, config.mapping_layer_size),
            torch.nn.LeakyReLU(0.2, inplace=True),
            stylegan_layers.EqualizedLinear(config.mapping_layer_size, config.mapping_layer_size),
            torch.nn.LeakyReLU(0.2, inplace=True),
            stylegan_layers.EqualizedLinear(config.mapping_layer_size, config.mapping_layer_size),
            torch.nn.LeakyReLU(0.2, inplace=True),
            stylegan_layers.EqualizedLinear(config.mapping_layer_size, config.style_size),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )

        def new_block(block_index):
            """Returns a block; we use a trick from the ProGAN paper to upscale and
            convolve at the same time in the first layer. We apply a conceptually
            similar trick to blur the image at the same time to get the bilinear
            upscale used in the paper without needing an extra tensor."""
            # If you modify the number or position of AdaIN layers in the model,
            # you'll need to update adaIN_layers and adaIN_offsets.
            self.adaIN_layers.append((3, 7))
            self.adaIN_offsets.append(self.adaIN_offsets[-1] + 2)
            return torch.nn.ModuleList([
                stylegan_layers.EqualizedConvTranspose2D(
                    in_channels=config.blocks[block_index - 1],
                    out_channels=config.blocks[block_index],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    upscale=True
                ),
                stylegan_layers.NoiseInput(config.blocks[block_index]),
                torch.nn.LeakyReLU(0.2, inplace=True),
                stylegan_layers.AdaIN(config.blocks[block_index], config.style_size),

                stylegan_layers.EqualizedConv2d(
                    in_channels=config.blocks[block_index],
                    out_channels=config.blocks[block_index],
                    kernel_size=3,
                    padding=1,
                ),
                stylegan_layers.NoiseInput(config.blocks[block_index]),
                torch.nn.LeakyReLU(0.2, inplace=True),
                stylegan_layers.AdaIN(config.blocks[block_index], config.style_size),
            ])

        # Block 0
        self.blocks.append(
            torch.nn.ModuleList([
                stylegan_layers.NoiseInput(config.blocks[0]),
                torch.nn.LeakyReLU(0.2, inplace=True),
                stylegan_layers.AdaIN(config.blocks[0], config.style_size),

                stylegan_layers.EqualizedConv2d(
                    in_channels=config.blocks[0],
                    out_channels=config.blocks[0],
                    kernel_size=3,
                    padding=1,
                ),
                stylegan_layers.NoiseInput(config.blocks[0]),
                torch.nn.LeakyReLU(0.2, inplace=True),
                stylegan_layers.AdaIN(config.blocks[0], config.style_size),
            ])
        )

        for block in range(len(config.blocks)):
            self.toRGBs.append(
                stylegan_layers.EqualizedConv2d(
                    in_channels=config.blocks[block],
                    out_channels=3,
                    kernel_size=1,
                    padding=0,
                )
            )

            # Don't add a new block on the last iteration, because the final block was
            # already here.
            if block < len(config.blocks) - 1:
                self.blocks.append(new_block(block + 1))

        # We need to register the blocks as modules for PyTorch to register their
        # weights as model parameters to optimize. We don't need to register adaINs
        # because they are already part of blocks
        self.toRGBs = torch.nn.ModuleList(self.toRGBs)
        self.blocks = torch.nn.ModuleList(self.blocks)

    def apply_block(self, features, block, block_id,
                    current_style, style_mappings, style_transitions):
        """
        Applies each layer in a block, and for AdaIN layers swap the current style
        mapping used by the generator if needed.
        """
        adaIN_count = 0
        for layer_id, layer in enumerate(block):
            if layer_id in self.adaIN_layers[block_id]:
                features = layer(features, current_style)
                # Next, perform style mixing by updating our current_style wherever
                # our style_transitions tensor says to.
                num_swaps = list(style_transitions.size())[1]
                if num_swaps < 1:  # Don't bother if no styles to mix
                    continue
                for alt_style in range(num_swaps):
                    # We'll have to check each style's slice of style_transitions to see
                    # if we update to that style. Note that we can't transition to
                    # style_mappings[0], the initial style, so while style_transitions
                    # just doesn't have space for the initial style, we have to skip
                    # ahead by 1 for indexing in style_mappings.
                    current_style = torch.where(
                        style_transitions[:,
                                          alt_style,
                                          adaIN_count + self.adaIN_offsets[block_id]],
                        style_mappings[alt_style + 1],
                        current_style,
                    )
                adaIN_count += 1
            else:
                features = layer(features)
        return features, current_style

    def forward(self, latent_samples, style_transitions, top_blocks, blend_ratio):
        features = self.ur_face

        lil_toRGB = self.toRGBs[top_blocks[0]]
        big_block = self.blocks[top_blocks[1]] if len(top_blocks) == 2 else None
        big_toRGB = self.toRGBs[top_blocks[1]] if len(top_blocks) == 2 else None

        style_mappings = [self.mapping_network(sample) for sample in latent_samples]
        current_style = style_mappings[0].clone()

        for i, block in enumerate(self.blocks):
            features, current_style = self.apply_block(features,
                                                       block,
                                                       i,
                                                       current_style,
                                                       style_mappings,
                                                       style_transitions)
            if i == top_blocks[0]:
                if len(top_blocks) == 1:
                    img = lil_toRGB(features)
                    return img
                else:
                    trained_img = lil_toRGB(features)
                    trained_img = torch.nn.functional.interpolate(trained_img,
                                                                  scale_factor=2.0,
                                                                  mode="nearest")
                    features, current_style = self.apply_block(features,
                                                               big_block,
                                                               top_blocks[1],
                                                               current_style,
                                                               style_mappings,
                                                               style_transitions)
                    new_img = big_toRGB(features)
                    return blend_ratio * new_img + (1.0 - blend_ratio) * trained_img

    def momentum_update(self, source_model, decay):
        """
        Updates the weights in self based on the weights in source_model.
        New weights will be decay * self's current weights + (1.0 - decay)
        * source_model's weights.

        This is used to make small updates to a visualizer network, moving each weight
        slightly toward the generator's weights. Doing it this way helps reduce
        artifacts and rapid changes in generated images, since we're averaging many
        states of the generator. The visualizer is what generates the sample images
        during training, and should probably be what's used for generation after
        training is complete.

        One thing worth noting is that images will appear less stable as training
        proceeds because, as the batch size shrinks, more updates to the visualizer are
        made between samples so this feature's effect on stability is diminished
        somewhat.
        """

        # Gets a dictionary mapping each generator parameter's name to the actual
        # parameter object. If you aren't using AccessibleDataParallel because you have
        # just one GPU, remove the "module" attribute and just use
        # dict(source_model.named_parameters())
        param_dict_src = dict(source_model.module.named_parameters())

        # For each parameter in the visualization model, get the same parameter in the
        # source model and perform the update.
        with torch.no_grad():
            for p_name, p_target in self.named_parameters():
                p_source = param_dict_src[p_name].to(p_target.device)
                p_target.copy_(decay * p_target + (1.0 - decay) * p_source)


class Discriminator(torch.nn.Module):
    """
    A discriminator between generated and real input images, as described in the paper.
    This model is composed of a set of blocks, each of which are trained in sequence.
    Block 0 is the last block data sees, outputting the discriminator's score.
    But Block 0 is also trained first.

    The final block computes the mean (across the entire feature map) standard deviation
    of pixel values (across the batch), and adds that as an extra feature to the input,
    then applies 1 convolution.
    Next, the block applies an unpadded 4x4 kernel to the resulting 4x4 featuremap,
    resulting in a 1x1 output (with 512 channels in the default configuration). A final
    convolution to a 1-channel output yields the discriminator's score of the input's
    "realness". This last convolution is equivalent to a "fully-connected" layer and is
    what the paper actually did in its code.

    As with the generator, each block contains a convolution layer plus a layer that
    applies both a convolution and downsample at the same time. The same leaky ReLU
    activation is used, but unlike the generator pixelwise normalization is not.

    Each block also has a fromRGB layer which converts an RGB image sized for that block
    into a featuremap with the number of channels expected by the block's first layer.
    """
    def __init__(self):
        super().__init__()

        self.blocks = []
        self.fromRGBs = []

        def new_block(block_index):
            return torch.nn.Sequential(
                stylegan_layers.EqualizedConv2d(
                    in_channels=config.blocks[block_index],
                    out_channels=config.blocks[block_index - 1],
                    kernel_size=3,
                    padding=1,
                ),
                torch.nn.LeakyReLU(0.2, inplace=True),

                stylegan_layers.EqualizedConv2d(
                    in_channels=config.blocks[block_index - 1],
                    out_channels=config.blocks[block_index - 1],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    downscale=True
                ),
                torch.nn.LeakyReLU(0.2, inplace=True),
            )

        self.blocks.append(
            torch.nn.Sequential(
                stylegan_layers.StandardDeviation(),
                stylegan_layers.EqualizedConv2d(
                    in_channels=config.blocks[0] + 1,  # +1 for std dev channel
                    out_channels=config.blocks[0],
                    kernel_size=3,
                    padding=1,
                ),
                torch.nn.LeakyReLU(0.2, inplace=True),

                # Input BxCx4x4; Output BxCx1x1
                stylegan_layers.EqualizedConv2d(
                    in_channels=config.blocks[0],
                    out_channels=config.blocks[0],
                    kernel_size=4,
                ),
                torch.nn.LeakyReLU(0.2, inplace=True),

                # Input BxCx1x1; collapsed to Bx1x1x1 - 1 score for each sample
                # With a little more refactoring, this could be replaced with a
                # stylegan_layers.EqualizedLinear().
                stylegan_layers.EqualizedConv2d(
                    in_channels=config.blocks[0], out_channels=1, kernel_size=1
                ),
            )
        )

        # We build the discriminator from back to front; this makes it a bit harder
        # to grok if you're examining debug info, since blocks[0] is the LAST block data
        # passes through, but it is considerably easier to read and follow the code.
        for block in range(len(config.blocks)):
            self.fromRGBs.append(
                stylegan_layers.EqualizedConv2d(
                    in_channels=3,
                    out_channels=config.blocks[block],
                    kernel_size=1,
                    padding=0,
                )
            )
            if block < len(config.blocks) - 1:
                self.blocks.append(new_block(block + 1))

        # As with the generator, convert our lists into ModuleLists to register them.
        self.blocks = torch.nn.ModuleList(self.blocks)
        self.fromRGBs = torch.nn.ModuleList(self.fromRGBs)

        # We'll also need a downscale layer for blending in training.
        self.halfsize = torch.nn.AvgPool2d(2)

        # Store some constants used in normalization of real samples
        self.mean = torch.nn.Parameter(torch.tensor([0.5, 0.5, 0.5]),
                                       requires_grad=False)
        self.std_dev = torch.nn.Parameter(torch.tensor([0.5, 0.5, 0.5]),
                                          requires_grad=False)

    def score_validity(self, img, top_blocks, blend_ratio):
        """
        This is the meat of the forward() method. The actual forward() method includes
        a few miscellaneous steps to support data preprocessing and the loss
        computation; it calls this when it needs to get a validity score for an input.
        """

        lil_fromRGB = self.fromRGBs[top_blocks[0]]
        big_block = self.blocks[top_blocks[1]] if len(top_blocks) == 2 else None
        big_fromRGB = self.fromRGBs[top_blocks[1]] if len(top_blocks) == 2 else None

        # The reverse of the generator - the layer we start training with depends on
        # which head is being trained, but we always proceed through to the end.
        if big_block is not None:
            features = big_fromRGB(img)
            features = big_block(features)
            trained_features = lil_fromRGB(self.halfsize(img))
            features = blend_ratio * features + (1.0 - blend_ratio) * trained_features
        else:
            features = lil_fromRGB(img)
        # The list slice here steps backward from the smaller-resolution of the top
        # blocks being trained
        for block in self.blocks[top_blocks[0]::-1]:
            features = block(features)

        # The view here just takes the output from Bx1x1x1 to B
        return features.view(-1)

    def forward(self, fake_img, real_imgs, top_blocks, blend_ratio):
        if real_imgs == None:
            # See below for a fuller discussion of the loss function. This is just the
            # fake image component of the same loss, used to train the generator.
            return torch.nn.functional.softplus(-1.0 *
                                                self.score_validity(fake_img,
                                                                    top_blocks,
                                                                    blend_ratio))
        else:
            real_imgs.requires_grad = True
            # Get the discriminator's opinion on a batch of fake images and one of real
            fake_validity = self.score_validity(fake_img, top_blocks, blend_ratio)
            real_validity = self.score_validity(real_imgs, top_blocks, blend_ratio)

            # This loss is equivalent to the non-saturating loss defined in the original
            # GAN paper, https://arxiv.org/abs/1406.2661. This minimizes the loss when
            # the discriminator is correct for both fake and real images, and has
            # mathematically convenient properties that keep the gradients from being
            # near-0 when the generator is not well-trained and the discriminator's job
            # is so easy it doesn't need to learn anything about the real data.

            # In the original paper, the discriminator output was assumed to be a number
            # between 0 and 1 - this is done by applying the sigmoid activation to the
            # output we already have. Then the optimization problem was to maximize
            # log(real_validity) + log(1 - fake_validity). For Pytorch, we need to
            # minimize rather than maximize, so we multiply that by -1. The softplus
            # implementation below is equivalent and faster to compute:
            # softplus(x) =  log(e^x  + 1) = -log(e^x/(e^x + 1)) = -log(sigmoid(x))

            fake_validity_loss = torch.nn.functional.softplus(fake_validity)
            real_validity_loss = torch.nn.functional.softplus(-1.0 * real_validity)

            # The gradient penalty penalizes the discriminator for having large
            # gradients on real images, discussed in https://arxiv.org/abs/1801.04406
            # For a rigorous explanation, see the paper - for intuition, the
            # discriminator is penalized for having gradients of any kind, which
            # discourages states that produce low-usefulness or spurious gradients,
            # because the gradient penalty needs to be outweighed by improved
            # performance on the task.

            real_gradients = torch.autograd.grad(
                outputs=torch.sum(real_validity),
                inputs=real_imgs,
                create_graph=True
            )[0]

            # I could not tell you why we scale the weight by 0.5, instead of choosing
            # the weight to be half of what we otherwise would - but I've left it this
            # way to make it easier to identify with equation 7 of the paper.
            gradient_penalty = config.gp_weight * 0.5 * torch.mean(
                torch.sum(real_gradients ** 2.0,
                          (1, 2, 3)  # Sum along channel, width, height dimensions
                          ),
            )

            return fake_validity_loss + real_validity_loss + gradient_penalty

