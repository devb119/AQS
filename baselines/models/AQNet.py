import torch
from torchvision.models import mobilenet_v3_small
import torch.nn as nn

def get_model(device, network,tabular_switch,S5p_switch, checkpoint=None):
    prediction_count = 1
    head_features = 96

    backbone_S2 = mobilenet_v3_small(pretrained=checkpoint, num_classes=1000)
    backbone_S2.features[0][0] = nn.Conv2d(12, 16, 3, 1, 1)
    backbone_S2.classifier[3] = nn.Linear(1024,S2_num_features)
    head_1 = nn.Sequential(
                    nn.Linear(S2_num_features+S5p_num_features, 384),
                    nn.ReLU(),
                    nn.Linear(384, 192),
                    nn.ReLU(),
                    nn.Linear(192, head_features))

    regression_model = RegressionHead_2(backbone_S2, backbone_S5P, head_2)

    # print("Displaying model architecture")
    # print(regression_model)

    return regression_model

class RegressionHead_2(nn.Module): # use s2, s5p
    def __init__(self, backbone_S5P, head_2):
        super(RegressionHead_2, self).__init__()
        self.backbone_S5P = backbone_S5P
        self.head_2 = head_2
    def forward(self, x):
        #get
        s5p = x.get("s5p")
        #backbones
        s5p = self.backbone_S5P(s5p)
        #satellites
        x = s5p
        out = self.head_2(x)
        return out