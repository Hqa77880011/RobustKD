import torch
from torch import nn
from torchvision.models import shufflenet_v2_x1_0


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.skip = (
            nn.Identity()
            if cin == cout and stride == 1
            else nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
            )
        )

    def forward(self, x):
        return (self.body(x) + self.skip(x)).relu()


class Teacher(nn.Module):
    def __init__(self, classes=100):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU()
        )
        layers = []
        cin = 64
        for stage, cout in enumerate([64, 128, 256]):
            for j in range(5):
                layers.append(Block(cin, cout, 2 if stage and j == 0 else 1))
                cin = cout
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(256, classes)
        self.features = 256

    def forward(self, x):
        f = self.body(self.stem(x)).mean((2, 3))
        return self.head(f), f


class Student(nn.Module):
    def __init__(self, classes=100):
        super().__init__()
        m = shufflenet_v2_x1_0(num_classes=classes)
        m.conv1[0].stride = (1, 1)
        self.body = nn.Sequential(m.conv1, m.stage2, m.stage3, m.stage4, m.conv5)
        self.head = m.fc
        self.features = m.fc.in_features

    def forward(self, x):
        f = self.body(x).mean((2, 3))
        return self.head(f), f


def cost(model):
    operations = []
    handles = []

    def hook(module, args, result):
        if isinstance(module, nn.Conv2d):
            operations.append(
                result.numel()
                * (module.in_channels // module.groups)
                * module.kernel_size[0]
                * module.kernel_size[1]
                * 2
            )
        else:
            operations.append(result.numel() * module.in_features * 2)

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32, device=next(model.parameters()).device))
    for h in handles:
        h.remove()
    return sum(p.numel() for p in model.parameters()), sum(operations)
