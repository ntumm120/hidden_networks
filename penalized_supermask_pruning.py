# General structure from https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class Supermask:
    def __init__(self, *args, **kwargs):
        self.sparsity = kwargs.get("sparsity", 1.0)
        kwargs.pop("sparsity", None)
        super().__init__(*args, **kwargs)
    
    def get_extra_state(self):
        return self.sparsity
    
    def set_extra_state(self, state):
        self.sparsity = state
    
    def freeze(self):
        self.scores.requires_grad = False
    
    def is_frozen(self):
        return not self.scores.requires_grad 
    
    def extra_repr(self):
        return f"{super().extra_repr()}, sparsity={self.sparsity}"

class SupermaskConv(Supermask, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        #self.sub_scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="sigmoid")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), self.sparsity)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class SupermaskLinear(Supermask, nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        self.calculate_subscores = False
        self.subscores = torch.zeros_like(self.scores).cuda() # to(self.scores.device)
        
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        
        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        
    def forward(self, x):
        if self.calculate_subscores:
            subnet = GetSubnet.apply(self.subscores.abs(), self.sparsity)
        else:
            subnet = GetSubnet.apply(self.scores.abs(), self.sparsity)
        
        subnet = subnet.cuda()
        w = self.weight * subnet
        
        return F.linear(x, w, self.bias)
        

# NOTE: not used here but we use NON-AFFINE Normalization!
# So there is no learned parameters for your nomralization layer.
class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)

def train(model, log_interval, device, train_loader, optimizer, criterion, epoch, penalty=0):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        model.fc1.subscores = torch.zeros_like(model.fc1.scores).to(model.fc1.scores.device)
        j = len(model.fc1.scores[0])
        model.fc1.subscores += ((((penalty / j  * torch.linalg.norm(model.fc2.scores.T, dim=1)).repeat(j, 1)).T) * torch.sign(model.fc1.scores))
        model.fc1.subscores += model.fc1.scores
        
        #print(torch.sum(model.fc1.subscores - model.fc1.scores))
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, criterion, test_loader, do_print=True, name="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    
    accuracy = 100. * correct / len(test_loader.dataset)
    loss = test_loss.item()
    if do_print:
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            name, test_loss, correct, len(test_loader.dataset),
            accuracy))
    return accuracy, loss
