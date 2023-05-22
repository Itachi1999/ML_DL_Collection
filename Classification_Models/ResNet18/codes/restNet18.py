import torch
import torch.nn as nn
import torchvision.models as models


class resNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=False) -> None:
        super().__init__()

        self.feature_extractor = models.resnet18(pretrained=pretrained)
        self.feature_extractor.fc = nn.Identity()
        self.model = nn.Sequential(
            self.feature_extractor,
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

        print(self.model)

    def forward(self, x):
        x = self.model(x)
        x = torch.softmax(x, dim=1)

        return x


def test():
    data = torch.randn(1, 3, 224, 224)
    model = resNet18()
    x = model(data)
    print(x.shape)
    print(x)


if __name__ == '__main__':
    test()
    # data1 = torch.randn(2)
    # data2 = torch.randn(2)

    # x = torch.cat((data1, data2))
    # print(x.shape)
