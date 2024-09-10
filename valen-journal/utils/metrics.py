from utils.utils_algo import to_logits
import torch
import torch.nn.functional as F


def evaluate(model, X, Y, device):
    with torch.no_grad():
        X, Y = map(lambda x: x.to(device), (X, Y))
        outputs = model(X)
        pred = to_logits(F.softmax(outputs, dim=1))
        acc = (pred * Y).sum() / Y.size(0)
    return acc.item()


def evaluate_benchmark(model, X, Y, device):
    with torch.no_grad():
        X, Y = map(lambda x: x.to(device), (X, Y))
        _, outputs = model(X)
        pred = to_logits(F.softmax(outputs, dim=1))
        acc = (pred * Y).sum() / Y.size(0)
    return acc.item()

def evaluate_realworld(model, X, Y, device):
    with torch.no_grad():
        X, Y = map(lambda x: x.to(device), (X, Y))
        outputs = model(X)
        pred = to_logits(F.softmax(outputs, dim=1))
        acc = (pred * Y).sum() / Y.size(0)
    return acc.item() * 100

def evaluate3(model, X, Y, device):
    with torch.no_grad():
        X, Y = map(lambda x: x.to(device), (X, Y))
        _, _, _, outputs, _ = model(X)
        pred = to_logits(outputs.clone().detach())
        acc = (pred * Y).sum() / Y.size(0)
    return acc.item()

def evaluate_gcn(model, gcn_X, test_Y, device):
    with torch.no_grad():
        gcn_X, test_Y = map(lambda x: x.to(device), (gcn_X, test_Y))
        outputs = model(gcn_X)
        pred = to_logits(F.softmax(outputs, dim=1))
        acc = (pred[-test_Y.size(0):] * test_Y).sum() / test_Y.size(0)
    return acc.item()
