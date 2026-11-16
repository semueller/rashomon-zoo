



import torch


from torch import nn
















class FCNN(nn.Module):
    def __init__(self, in_features, feature_multiplier=1, classes=2):
        super(FCNN, self).__init__()
        torch.manual_seed(1234)
        self.name = "fcnn"
        self.fc1 = nn.Linear(in_features=in_features, out_features=int(in_features * feature_multiplier))
        self.fc2 = nn.Linear(in_features=int(in_features * feature_multiplier),
                             out_features=int(in_features * feature_multiplier))
        self.fc3 = nn.Linear(in_features=int(in_features * feature_multiplier), out_features=classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def predict_batch_softmax(self, x):
        x = self(x)
        return self.softmax(x)


def make_ff(shapes_layers, actfun_out=None):
    # shapes layers: output size layer is input size of next ..
    layers = [nn.Linear(s_in, s_out) for (s_in, s_out) in zip(shapes_layers[:-1], shapes_layers[1:])]
    actfun = nn.ReLU
    architecture = []
    for layer in layers:
        architecture.append(layer)
        architecture.append(actfun())
    architecture = architecture[:-1] # delete last actfun
    if actfun_out is not None:
        architecture.append(actfun())
    # architecture2 = [
    #     nn.Linear(16,16),
    #     nn.ReLU(),
    #     nn.Linear(16,16),
    #     nn.ReLU(),
    #     nn.Linear(16,2),
    # ]
    sequential = nn.Sequential(*architecture)
    return SimpleNet(sequential=sequential)


class SimpleNet(nn.Module):
    def __init__(self, sequential):
        super(SimpleNet, self).__init__()
        self.net = sequential
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.net(x)
        return x

    def predict_batch(self, x):
        pred = self.forward(x)
        return torch.argmax(pred, dim=1)

    def predict_batch_softmax(self, x):
        pred = self.forward(x)
        sm_pred = self.softmax(pred)
        return sm_pred


