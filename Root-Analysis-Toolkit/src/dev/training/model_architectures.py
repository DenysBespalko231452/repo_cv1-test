import torch
import torch.nn as nn
from torch.utils.data import Dataset


class UNetModel(nn.Module):
    """
    Flexible PyTorch implementation of the U-Net architecture.

    This model supports configurable input size, three predefined model
    scales (small, medium, large), optional batch normalization, and
    the ability to freeze encoder and/or decoder layers during training.

    Attributes:
        SUPPORTED_SIZES (set): Allowed spatial dimensions for input images.
        SIZE_TO_FILTERS (dict): Maps model_size strings to base filter counts.

    Args:
        dataset (Dataset, optional): PyTorch Dataset to infer in_channels,
            height, and width from sample[0]. If provided, overrides
            in_channels, height, and width arguments.
        in_channels (int, optional): Number of input channels (e.g., 3 for RGB).
        height (int, optional): Height of input images (must equal width).
        width (int, optional): Width of input images (must equal height).
        model_size (str): One of {'small','medium','large'}. Determines
            the base number of convolutional filters used (powers of two).
        use_batch_norm (bool): If True, include BatchNorm layers after each
            convolution for better training stability.
        freeze_encoder (bool): If True, encoder layers (downsampling path)
            will have requires_grad=False.
        freeze_decoder (bool): If True, decoder layers (upsampling path)
            and final output conv will have requires_grad=False.

    Raises:
        ValueError: If model_size is invalid, or if neither dataset nor
            explicit dimensions are provided, or if spatial dims are
            not equal or unsupported.

    Example:
        # Infer size from a dataset, use medium-scale, batch norm,
        # and freeze the encoder during fine-tuning:
        model = UNetModel(
            dataset=my_dataset,
            model_size='medium',
            use_batch_norm=True,
            freeze_encoder=True
        )

        # Explicitly define a small-scale U-Net on 256×256 RGB images:
        model = UNetModel(
            in_channels=3,
            height=256,
            width=256,
            model_size='small',
            use_batch_norm=False
        )
    """
    SUPPORTED_SIZES = {64, 128, 256, 512}
    SIZE_TO_FILTERS = {
        'small': 1,
        'medium': 2,
        'large': 3,
    }

    def __init__(
        self,
        dataset: Dataset = None,
        in_channels: int = None,
        height: int = None,
        width: int = None,
        model_size: str = 'small',
        use_batch_norm: bool = False,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
    ):
        super().__init__()

        # Validate chosen model_size and derive base filter multiplier
        if model_size not in self.SIZE_TO_FILTERS:
            allowed = list(self.SIZE_TO_FILTERS)
            raise ValueError(
                f"model_size must be one of {allowed}, got '{model_size}'"
            )
        base_filters = self.SIZE_TO_FILTERS[model_size]
        self.use_bn = use_batch_norm

        # Infer input dimensions from dataset if provided
        
        if dataset is not None:
            sample = dataset[0][0]
            if not isinstance(sample, torch.Tensor):
                raise ValueError("Dataset samples must be torch.Tensor images")

            # If sample is H×W×C (e.g. from a HWC dataset), move channels to front
            if sample.ndim == 3 and sample.shape[2] in (1, 3):
                sample = sample.permute(2, 0, 1).contiguous()

            in_channels, height, width = sample.shape

        # Ensure explicit dimensions are provided if dataset was None
        if any(v is None for v in (in_channels, height, width)):
            raise ValueError(
                "Provide either a dataset or explicit "
                "in_channels, height, and width"
            )

        # Enforce square inputs of supported size
        if height != width or height not in self.SUPPORTED_SIZES:
            dims = f"{height}x{width}"
            supported = sorted(self.SUPPORTED_SIZES)
            raise ValueError(
                f"Spatial dimensions must be equal and one of {supported}, "
                f"got {dims}"
            )

        # Build encoder (downsampling) blocks
        f = base_filters
        self.enc1 = self._conv_block(in_channels, f, dropout=0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2 = self._conv_block(f, f * 2, dropout=0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3 = self._conv_block(f * 2, f * 4, dropout=0.2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4 = self._conv_block(f * 4, f * 8, dropout=0.2)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5 = self._conv_block(f * 8, f * 16, dropout=0.3)
        self.pool5 = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = self._conv_block(f * 16, f * 32, dropout=0.3)

        # Build decoder (upsampling) blocks
        self.up6 = nn.ConvTranspose2d(
            f * 32, f * 16, kernel_size=2, stride=2
        )
        self.dec6 = self._conv_block(f * 32, f * 16, dropout=0.2)

        self.up7 = nn.ConvTranspose2d(
            f * 16, f * 8, kernel_size=2, stride=2
        )
        self.dec7 = self._conv_block(f * 16, f * 8, dropout=0.2)

        self.up8 = nn.ConvTranspose2d(
            f * 8, f * 4, kernel_size=2, stride=2
        )
        self.dec8 = self._conv_block(f * 8, f * 4, dropout=0.1)

        self.up9 = nn.ConvTranspose2d(
            f * 4, f * 2, kernel_size=2, stride=2
        )
        self.dec9 = self._conv_block(f * 4, f * 2, dropout=0.1)

        self.up10 = nn.ConvTranspose2d(
            f * 2, f, kernel_size=2, stride=2
        )
        self.dec10 = self._conv_block(f * 2, f, dropout=0.1)

        # Final 1x1 convolution to map to single-channel output
        self.out_conv = nn.Conv2d(in_channels=f, out_channels=1, kernel_size=1)

        # Optionally freeze encoder parameters
        if freeze_encoder:
            for name, param in self.named_parameters():
                if name.startswith('enc') or name.startswith('pool'):
                    param.requires_grad = False

        # Optionally freeze decoder parameters and output layer
        if freeze_decoder:
            for name, param in self.named_parameters():
                if name.startswith('up') or name.startswith('dec') or (
                        'out_conv' in name):
                    param.requires_grad = False

    def _conv_block(
        self, in_ch: int, out_ch: int, dropout: float
    ) -> nn.Sequential:
        """
        Creates a two-layer convolutional block:
          - Conv2d(in_ch → out_ch, 3×3, padding=1)
          - Optional BatchNorm (+ReLU)
          - Conv2d(out_ch → out_ch, 3×3, padding=1)
          - Optional BatchNorm (+ReLU)
          - Dropout

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            dropout (float): Dropout probability after convolutions.

        Returns:
            nn.Sequential: The stacked layers as a sequential module.
        """
        layers = [
            nn.Conv2d(
                in_channels=in_ch, out_channels=out_ch,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(out_ch) if self.use_bn else None,
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_ch, out_channels=out_ch,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(out_ch) if self.use_bn else None,
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        ]
        # Filter out any None entries (i.e., if batch norm disabled)
        return nn.Sequential(*(layer for layer in layers if layer))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net:
          1. Downsample path: enc1→pool1 → enc2→pool2 ... → enc5→pool5
          2. Bottleneck convolution
          3. Upsample path with skip connections:
             up6+dec6 (skip c5), up7+dec7 (skip c4), ..., up10+dec10 (skip c1)
          4. Final 1×1 conv and sigmoid activation

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output mask of shape
                (batch_size, 1, height, width), with values in [0,1].
        """
        # Encoder: capture intermediate feature maps for skip connections
        c1 = self.enc1(x)
        p1 = self.pool1(c1)

        c2 = self.enc2(p1)
        p2 = self.pool2(c2)

        c3 = self.enc3(p2)
        p3 = self.pool3(c3)

        c4 = self.enc4(p3)
        p4 = self.pool4(c4)

        c5 = self.enc5(p4)
        p5 = self.pool5(c5)

        # Bottleneck feature representation
        bn = self.bottleneck(p5)

        # Decoder: upsample and concatenate skip connections
        u6 = self.up6(bn)
        c6 = self.dec6(torch.cat([u6, c5], dim=1))

        u7 = self.up7(c6)
        c7 = self.dec7(torch.cat([u7, c4], dim=1))

        u8 = self.up8(c7)
        c8 = self.dec8(torch.cat([u8, c3], dim=1))

        u9 = self.up9(c8)
        c9 = self.dec9(torch.cat([u9, c2], dim=1))

        u10 = self.up10(c9)
        c10 = self.dec10(torch.cat([u10, c1], dim=1))

        # Final convolution and activation
        out = self.out_conv(c10)
        return torch.sigmoid(out)
