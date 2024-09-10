import torch
import torchvision
from torch import nn


class Resnet34(nn.Module):
    def __init__(self, num_classes):
        super(Resnet34, self).__init__()
        self.model = torchvision.models.resnet34(pretrained=False, progress=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.model.avgpool.register_forward_hook(self._get_activation('avgpool'))
        self.activation = {}

    def _get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def forward(self, x):
        output = self.model(x)
        phi = torch.flatten(self.activation['avgpool'], 1)
        return phi, output
        # return 0, output

# model = Resnet34(5)
# X1 = torch.rand((2, 3, 224, 224))
# X2 = torch.rand((2, 3, 224, 224))
# phi, out = model(X1)
# phi = torch.flatten(phi, 1)
# # print(model)
# # print(phi.size())
# print(phi)
# # print(out)
# phi, out = model(X2)
# phi = torch.flatten(phi, 1)
# # print(model)
# # print(phi.size())
# print(phi)
# # print(out)