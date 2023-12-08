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
        if not isinstance(k, float):
            threshold, min_sparsity = k
            unpruned = scores > threshold
            if unpruned.sum() / scores.numel() > min_sparsity:
                return unpruned.int()
            else:
                k = min_sparsity
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
        self.sparsity = kwargs.pop("sparsity", None)
        self.min_sparsity = kwargs.pop("min_sparsity", None)
        self.threshold = kwargs.pop("threshold", None)
        self.init = kwargs.pop("init", None)
        assert (self.sparsity is not None) ^ (self.min_sparsity is not None and self.threshold is not None)
        
        if self.sparsity is not None:
            self.main_sparsity = self.sparsity
        else:
            self.main_sparsity = (self.threshold, self.min_sparsity)
        
        super().__init__(*args, **kwargs)
        
        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        # nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        self.initialize()

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        
        if self.bias is not None:
            print("Using biases which may not be initialized correctly!")
            self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
            # nn.init.kaiming_uniform_(self.bias_scores, a=math.sqrt(5))
            # nn.init.kaiming_normal_(self.bias, mode='fan_out', nonlinearity="relu")
            self.bias.requires_grad = False
    
    def initialize(self):
        
        k = self.sparsity if self.sparsity is not None else self.min_sparsity
        mode = "fan_in"
        scale_fan = False
        
        if self.init == "signed_constant":

            fan = nn.init._calculate_correct_fan(self.weight, mode)
            if scale_fan:
                fan = fan * (1 - k)
            # gain = nn.init.calculate_gain("relu")
            std = 1 / math.sqrt(fan)
            self.weight.data = self.weight.data.sign() * std

        elif self.init == "unsigned_constant":

            fan = nn.init._calculate_correct_fan(self.weight, mode)
            if scale_fan:
                fan = fan * (1 - k)

            # gain = nn.init.calculate_gain("relu")
            std = 1 / math.sqrt(fan)
            self.weight.data = torch.ones_like(self.weight.data) * std

        elif self.init == "kaiming_normal":

            if scale_fan:
                fan = nn.init._calculate_correct_fan(self.weight, mode)
                fan = fan * (1 - k)
                gain = nn.init.calculate_gain("relu")
                std = gain / math.sqrt(fan)
                with torch.no_grad():
                    self.weight.data.normal_(0, std)
            else:
                nn.init.kaiming_normal_(
                    self.weight, mode=mode, nonlinearity="relu"
                )

        elif self.init == "kaiming_uniform":
            nn.init.kaiming_uniform_(
                self.weight, mode=mode, nonlinearity="relu"
            )
        elif self.init == "xavier_normal":
            nn.init.xavier_normal_(self.weight)
        elif self.init == "xavier_constant":

            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            self.weight.data = self.weight.data.sign() * std

        elif self.init == "standard":

            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        else:
            raise ValueError(f"{self.init} is not an initialization option!")
        
            
    def get_extra_state(self):
        return self.sparsity
    
    def set_extra_state(self, state):
        self.sparsity = state
    
    def freeze(self):
        self.scores.requires_grad = False
    
    def is_frozen(self):
        return not self.scores.requires_grad 
    
    def extra_repr(self):
        if self.sparsity is None:
            return f"{super().extra_repr()}, threshold={self.threshold},min_sparsity={self.min_sparsity}"
        else:
            return f"{super().extra_repr()}, sparsity={self.sparsity}"

class SupermaskConv(Supermask, nn.Conv2d):
    
    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), self.main_sparsity)
        w = self.weight * subnet
        if self.bias is not None:
            subnet_bias = GetSubnet.apply(self.bias.abs(), self.main_sparsity)
            b = self.bias * subnet_bias
        else:
            b = self.bias
        x = F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class SupermaskLinear(Supermask, nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subscores = torch.zeros_like(self.scores).cuda() # to(self.scores.device)
        
    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), self.main_sparsity)
        subnet = subnet.cuda()
        w = self.weight * subnet
        if self.bias is not None:
            subnet_bias = GetSubnet.apply(self.bias.abs(), self.main_sparsity)
            b = self.bias * subnet_bias
        else:
            b = self.bias
        return F.linear(x, w, b)
        return x

# NOTE: not used here but we use NON-AFFINE Normalization!
# So there is no learned parameters for your nomralization layer.
class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SupermaskConv(1, 32, 3, 1, bias=False)
        self.conv2 = SupermaskConv(32, 64, 3, 1, bias=False)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = SupermaskLinear(9216, 128, bias=False)
        self.fc2 = SupermaskLinear(128, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, log_interval, device, train_loader, optimizer, criterion, epoch, penalty=0):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        model.fc1.subscores = torch.zeros_like(model.fc1.scores).to(model.fc1.scores.device)
        j = len(model.fc1.scores[0])
        model.fc1.subscores += ((((penalty / j  * torch.linalg.norm(model.fc2.scores.T, dim=1)).repeat(j, 1)).T) * torch.sign(model.fc1.scores))
        
        # print("Scores")
        # print(model.fc1.subscores)
        # print(torch.unique(torch.abs(model.fc1.subscores)))
        #print(torch.sum(model.fc1.subscores - model.fc1.scores))
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        #print(model.fc1.scores)
        with torch.no_grad():
            model.fc1.scores += model.fc1.subscores
        
        #print(model.fc1.scores)
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
