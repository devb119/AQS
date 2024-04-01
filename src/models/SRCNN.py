import math
import torch
from torch import nn

class SRCNN(nn.Module):
    def __init__(self) -> None:
        super(SRCNN, self).__init__()
        # Adjusted feature extraction layer for 11-channel input.
        self.features = nn.Sequential(
            nn.Conv2d(11, 64, (9, 9), (1, 1), (4, 4)),  # Adjusted for 11-channel input
            nn.ReLU(True)
        )

        # Non-linear mapping layer remains unchanged.
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True)
        )

        # Rebuild the layer, adjusted for 11-channel output.
        self.reconstruction = nn.Conv2d(32, 11, (5, 5), (1, 1), (2, 2))  # Adjusted for 11-channel output

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)
        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Adjusted weight initialization considering the layer's fan-in
                nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)
        # Specifically handling the reconstruction layer's initialization
        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)
