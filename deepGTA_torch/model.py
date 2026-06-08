import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.stddev
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with Conv2d layers.

    FIX #5: Added BatchNorm2d after every convolution.
    The original architecture had no normalisation at all, which caused
    unstable training loss and slow convergence.

    Pattern per conv: Conv → BN → ReLU  (pre-activation style on shortcut too)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 num_operations, project_identity=True):
        super().__init__()

        if isinstance(kernel_size, int):
            pad = kernel_size // 2
        else:
            pad = (kernel_size[0] // 2, kernel_size[1] // 2)

        layers = []
        for i in range(num_operations - 1):
            ch_in = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(ch_in, out_channels, kernel_size, padding=pad, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))   # FIX #5
            layers.append(nn.ReLU(inplace=True))

        # Last conv — no ReLU here; activation comes after the residual add
        ch_in = in_channels if num_operations == 1 else out_channels
        layers.append(nn.Conv2d(ch_in, out_channels, kernel_size, padding=pad, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))        # FIX #5
        self.convs = nn.Sequential(*layers)

        if project_identity or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),              # FIX #5
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.convs(x)
        res = self.shortcut(x)
        return F.relu(out + res, inplace=True)


class DeepSKAN(nn.Module):
    """
    Deep Spectroscopy Kinetic Analysis Network.

    Changes vs original:
      FIX #5  — BatchNorm2d in every residual block (training stability).
      FIX #6  — Dropout before FC layers (regularisation, sim→real gap).
      FIX #7  — AdaptiveAvgPool2d replaces hardcoded flatten size of 2048,
                 so the model is robust to input resolution changes.
    """

    def __init__(self, num_classes: int = 103, dropout_p: float = 0.3):
        super().__init__()

        self.noise = GaussianNoise(0.01)

        # ── Convolutional backbone ────────────────────────────────────────────
        # Blocks 1-2: tall asymmetric kernels → capture time-domain structure
        self.res1  = ResidualBlock(1,   64,  kernel_size=(15, 1), num_operations=4)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))

        self.res2  = ResidualBlock(64,  128, kernel_size=(11, 1), num_operations=4)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))

        # Blocks 3-4: start mixing time + wavelength dimensions
        self.res3  = ResidualBlock(128, 128, kernel_size=(9, 3),  num_operations=4, project_identity=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.res4  = ResidualBlock(128, 128, kernel_size=(9, 3),  num_operations=4, project_identity=False)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        # Blocks 5-7: square kernels — spatial feature extraction
        self.res5  = ResidualBlock(128, 256, kernel_size=(5, 5),  num_operations=4)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2))

        self.res6  = ResidualBlock(256, 256, kernel_size=(3, 3),  num_operations=4, project_identity=False)
        self.pool6 = nn.MaxPool2d(kernel_size=(2, 2))

        self.res7  = ResidualBlock(256, 128, kernel_size=(3, 3),  num_operations=2)

        # FIX #7: AdaptiveAvgPool collapses spatial dims to a fixed (2, 4) map
        # giving 128 * 2 * 4 = 1024 features regardless of input resolution.
        # This replaces the brittle hardcoded nn.Linear(2048, ...).
        self.global_pool = nn.AdaptiveAvgPool2d((2, 4))

        # ── Classifier head ───────────────────────────────────────────────────
        # FIX #6: Dropout before each FC layer for regularisation.
        self.dropout = nn.Dropout(p=dropout_p)

        self.fc1 = nn.Linear(128 * 2 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512,  512)
        self.fc4 = nn.Linear(512,  256)
        self.fc5 = nn.Linear(256,  num_classes)

    def forward(self, x):
        x = self.noise(x)

        x = self.pool1(self.res1(x))
        x = self.pool2(self.res2(x))
        x = self.pool3(self.res3(x))
        x = self.pool4(self.res4(x))
        x = self.pool5(self.res5(x))
        x = self.pool6(self.res6(x))
        x = self.res7(x)

        # FIX #7: adaptive pool then flatten — no hardcoded size
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        # FIX #6: dropout between FC layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.fc5(x)   # raw logits — loss function applies softmax internally

        return x
