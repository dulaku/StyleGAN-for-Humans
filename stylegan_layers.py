import torch
import numpy

class Pixnorm(torch.nn.Module):
    """
    Normalizes the input tensor to magnitude 1 without changing direction;
    as a different perspective, maps inputs to points on an N-dimensional sphere
    with radius 1, where N is the number of input dimensions.
    """
    def forward(self, input):
        channels = input.size()[1] # Inputs are batch x channel x height x width
        normalizer = torch.sqrt(
            1e-8 + torch.sum(input ** 2.0, dim=1, keepdim=True) / channels
        )
        return input.div(normalizer)


class StandardDeviation(torch.nn.Module):
    """
    Adds an extra channel to the input, containing the standard deviation of
    pixel values across the batch. The new channel contains the mean of each
    pixel's standard deviation. This gives the discriminator a clue to detect
    generated batches if the generator only learns how to generate one image,
    which forces the generator to learn a distribution of images to generate.
    """
    def forward(self, input):
        batch_size, _, height, width = input.shape

        # B x C x H x W; Difference from the mean at each location
        output = input - input.mean(dim=0, keepdim=True)

        # C x H x W; Standard deviation at each location across the batch
        output = torch.sqrt_(output.pow_(2.0).mean(dim=0, keepdim=False) + 10e-8)

        # 1 x 1 x 1 x 1; Mean standard deviation across entire featuremap
        output = output.mean().view(1, 1, 1, 1)

        # B x 1 x H x W; Copy the mean to all locations of a new channel
        output = output.repeat(batch_size, 1, height, width)

        # Append that channel to the original input
        output = torch.cat([input, output], 1)
        return output


class EqualizedConv2d(torch.nn.Module):
    """
    This is mainly a standard convolutional layer, but with the added feature
    that its weights are scaled by the constant given in He 
    (https://arxiv.org/abs/1502.01852). The variance in gradients for a layer
    depends on the size of its input; scaling to account for this helps smooth
    out training and helps ensure the model converges to something. 

    The linked paper only applies scaling at initialization; this layer instead
    initializes without scaling, and does the scaling at every forward pass. This still
    helps account for the variance during training, but since the weight's value isn't
    changed by scaling, a large scaling factor won't reduce the value to nearly 0 and
    thus won't ensure the gradients with respect to that weight are nearly 0 no matter
    what the loss happens to be.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0,
        downscale=False
    ):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.downscale = downscale
        self.in_channels = in_channels

        # Create an empty tensor for the filter's weights, then initialize it
        self.weights = torch.nn.Parameter(
            torch.nn.init.normal_(  # Underscore suffix for in-place operation
                torch.empty(out_channels, in_channels, kernel_size, kernel_size)
            ),
            requires_grad=True,
        )

        self.bias = torch.nn.Parameter(torch.FloatTensor(out_channels).fill_(0))

        # He initialization scaling factor
        fan_in = kernel_size * kernel_size * in_channels
        self.scale = numpy.sqrt(2) / numpy.sqrt(fan_in)

        if self.downscale:
            # In this case, we need to prepare a few constants so we can blur the input
            # and wind up with a bilinear downsample. We'll do this by constructing a
            # bigger kernel to apply that includes both the blur and the learned
            # convolution in one step.
            gaussian_kernel = numpy.array(
                [[[[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]]]]
            )
            gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
            self.gaussian_kernel = torch.nn.Parameter(
                torch.FloatTensor(
                    numpy.broadcast_to(gaussian_kernel, (in_channels, 1, 3, 3))
                ),
                requires_grad=False
            )

            # As it turns out, our larger kernel is going to need extra padding. To keep
            # the calling signature of this layer the same, since the convolution we're
            # actually learning still has the same size, stride, and padding, we're
            # hiding the change here instead of just calling the layer with a bigger
            # padding value.
            self.padding += 1

    def forward(self, input):
        if self.downscale:
            # Pad the last two dimensions (HxW) of the kernel weights
            weight = torch.nn.functional.pad(self.weights, [1, 1, 1, 1])
            # Blur the weights by averaging the 4 4x4 corners; if we didn't do this
            # every second weight-pixel pair would be skipped with the stride
            # of 2.
            weight = (  weight[:, :,  :-1,  :-1]
                      + weight[:, :,  :-1, 1:]
                      + weight[:, :, 1:,    :-1]
                      + weight[:, :, 1:,   1:]) / 4.0

            # The above doesn't account for the blur in bilinear downsampling - the
            # below steps apply it without a separate conv layer by composing the
            # weights kernel with a Gaussian blur kernel. The end result is a 6x6 kernel
            # we apply to the base input with a stride of 2 and padding of 2. This
            # behaves like blurring before rescaling, which is what we want. Note that,
            # technically, this introduces a small error along the 1-pixel border of the
            # image - in practice, the model seems to deal reasonably well with this.
            # You can get a strictly correct implementation by removing this logic, in
            # other words by replacing this class with the ProGAN implementation of the
            # same class, and adding a Gaussian blur layer immediately before this layer
            # in the model. You can see an example blur layer below.
            weight = torch.nn.functional.conv2d(
                input=weight,
                weight=self.gaussian_kernel,
                stride=1,
                padding=2,
                groups=self.in_channels
            )

        else:
            weight = self.weights
        return torch.nn.functional.conv2d(
            input=input,
            weight=weight * self.scale,  # Scaling weights dynamically
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )


class EqualizedConvTranspose2D(torch.nn.Module):
    """
    This is a transpose convolution layer, modified to support dynamic scaling
    in the same way as the above layer. For an in-depth guide to the behavior of
    transposed convolutions, https://arxiv.org/pdf/1603.07285v1.pdf is a good source
    for some intuition, as is
    https://medium.com/apache-mxnet/transposed-convolutions-explained-with-ms-excel-52d13030c7e8
    but for us the key takeaway is that transpose convolution behaves like a
    typical convolution, but padded with zeros in specific ways.

    Adding strides to a transpose convolution spreads where the kernel is applied to the
    output, rather than the input - that is, you still apply the kernel to each input
    pixel, but when you move from one input pixel to the next, the kernel's output
    in the output image will skip several pixels instead of 1. This upscales the output.

    Additionally, "padding" a transpose convolution means discarding data from the edges
    of the output, rather than adding empty data to the edges of the input. With no
    padding, you get a larger output even without striding (that's how
    an unpadded 1x1 input from our latent space is converted to a starting 4x4), while
    the traditional amount of padding to keep the output the same size in a regular
    convolution does the same thing in a transpose convolution.

    There are gotchas with transpose convolutions that can lead to bad artifacting in
    generated images, so if you want to tweak the kernel sizes, I recommend delving into
    the math a bit. https://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, upscale=False):
        super().__init__()

        self.weights = torch.nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(in_channels, out_channels, kernel_size, kernel_size)
            )
        )

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.out_channels = out_channels

        self.bias = torch.nn.Parameter(torch.FloatTensor(out_channels).fill_(0))

        # Fan-in is simply number of input channels
        self.scale = numpy.sqrt(2) / numpy.sqrt(in_channels)

        self.upscale = upscale

        if self.upscale:
            # Same as for downsampling, we need to prepare a few constants so we can
            # blur the input and wind up with a bilinear upsample. We'll do this by
            # constructing a bigger kernel to apply that includes both the blur and the
            # learned convolution.
            gaussian_kernel = numpy.array(
                [[[[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]]]]
            )
            gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
            self.gaussian_kernel = torch.nn.Parameter(
                torch.FloatTensor(
                    numpy.broadcast_to(gaussian_kernel, (out_channels, 1, 3, 3))
                ),
                requires_grad=False
            )

            self.padding += 1

    def forward(self, input):
        if self.upscale:
            # Pad the last two dimensions (HxW) of the kernel weights
            weight = torch.nn.functional.pad(self.weights, [1, 1, 1, 1])

            # Blur the weights by summing the 4 4x4 corners; this again ensures that
            # each weight-pixel pair is involved in computing the outcome since we have
            # that pesky stride of 2 to downsample. The way the algebra works out, we
            # do not want to divide by 4.0 here.
            weight = (  weight[:, :,  :-1,  :-1]
                      + weight[:, :,  :-1, 1:]
                      + weight[:, :, 1:,    :-1]
                      + weight[:, :, 1:,   1:])

            # Just as with the downsampling blur, we'll compose the kernels by
            # convolving our learned kernel with the gaussian. Rather conveniently,
            # the symmetry between transpose and standard convolution means that this
            # behaves like blurring after rescaling, which is again what we want.

            # The paper is somewhat misleading - while the source code uses blurring
            # when downscaling, the paper claims to use bilinear interpolation in its
            # downscale operations - but that's what we'd be doing if we _didn't_ blur.
            # The blur is only necessary to make _upscaling_ equivalent to bilinear
            # interpolation, but it seems to work well here so we're leaving it in.
            # Try comparing to the output of torch.resize() with mode='bilinear'.

            weight = torch.nn.functional.conv2d(
                input=weight,
                weight=self.gaussian_kernel,
                stride=1,
                padding=2,
                groups=self.out_channels
            )

        else:
            weight = self.weights
        return torch.nn.functional.conv_transpose2d(
            input=input,
            weight=weight * self.scale,  # Dynamic weight scaling
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )


class EqualizedLinear(EqualizedConv2d):
    """
    A fully-connected layer (AKA perceptron, AKA feed-foward layer), supporting the same
    weight scaling trick as above. In fact, it's a Conv2d layer under the hood - a 1x1xN
    input with a 1x1 kernel and M output channels is equivalent to an FC layer mapping a
    size-N vector to size-M, a fact we exploited in ProGAN. We use these a lot more in
    StyleGAN, so it deserves a separate class for clarity in reading.
    """
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size, kernel_size=1)

    def forward(self, input):
        # Add 2 size-1 dimensions to the input, then remove them before returning
        return super().forward(input[:, :, None, None])[:, :, 0, 0]


class AdaIN(torch.nn.Module):
    """
    Adaptive Instance Normalization combines two techniques. First, each input is
    normalized independently. You can think of this as adjusting the image contrast to
    a fixed value, which ensures the magnitude of the input's values are reasonable
    without destroying any information (instance normalization).

    Unlike batch normalization, this computation isn't done across example data in a
    batch - this means that some images can have more influence on training than others
    just by having more high-intensity features, but it also removes a source of noise
    in training, since training doesn't depend on which examples appear in a batch
    together. It also still provides the needed normalization to keep the network stable
    that Pixnorm did, but with more flexibility.

    After instance normalization, each channel is scaled and has a bias added. This is
    the "style" - the model learns to select scaling factors and biases to exaggerate
    or suppress different features in the input. This ultimately depends on the sample
    from the latent space, which instructs the model as to which features to manipulate.
    """
    def __init__(self, in_channels, style_size):
        super().__init__()
        self.instance_norm = torch.nn.InstanceNorm2d(in_channels)

        # We need to initialize the bias to 1 so that, on average, the output at the
        # start of training will be 1; since features will be multiplied by this layer's
        # outputs, this avoids drastic changes until the model learns to make helpful
        # ones.
        self.weight_layer = EqualizedLinear(style_size, in_channels)
        self.weight_layer.bias.data = torch.nn.Parameter(
            torch.ones_like(self.weight_layer.bias.data)
        )

        # The default bias of 0 is fine here, since we add these outputs, not multiply
        self.bias_layer = EqualizedLinear(style_size, in_channels)

    def forward(self, in_features, style_mapping):
        # The weights and biases need size-1 height and width dimensions added to them
        # for the operations to follow
        style_weights = self.weight_layer(style_mapping)[:, :, None, None]
        style_bias = self.bias_layer(style_mapping)[:, :, None, None]
        normalized = self.instance_norm(in_features)

        return normalized * style_weights + style_bias

class NoiseInput(torch.nn.Module):
    """
    Applies weight to an input with learned scaling factor for the noise applied to each
    channel. This adds random variation to styles, which allows the generator to not
    need to learn how to do that.
    """
    def __init__(self, channels):
        super().__init__()
        # We initialize this to 0 so that no noise is added until the model learns to
        # add it to handle randomness in images.
        self.weights = torch.nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, input):
        return input + self.weights * torch.randn_like(input)


class Blur(torch.nn.Module):
    """
    Applies a Gaussian blur to inputs using a 2D convolution.
    """
    def __init__(self, channels):
        super().__init__()
        gaussian_kernel = numpy.array(
            [[[[1, 2, 1],
               [2, 4, 2],
               [1, 2, 1]]]]
        )
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        self.gaussian_kernel = torch.nn.Parameter(
            torch.FloatTensor(
                numpy.broadcast_to(gaussian_kernel, (channels, 1, 3, 3))
            ),
            requires_grad=False
        )
        self.channels = channels

    def forward(self, in_features):
        blurred = torch.nn.functional.conv2d(
            input=in_features,
            weight=self.gaussian_kernel,
            stride=1,
            padding=1,
            groups=self.channels
        )
        return blurred
