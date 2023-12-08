import matplotlib.pyplot as plt
import os
os.environ['HOME_DIR'] = 'drive/MyDrive/hidden-networks'

import sys
sys.path.append(os.path.join('/content', os.environ['HOME_DIR']))
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd
import collections
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.stats import hypsecant
from scipy.stats import uniform, norm
from torch.autograd.functional import jacobian

from signal_supermask_pruning2 import GetSubnet, SupermaskConv, SupermaskLinear
from signal_supermask_pruning2 import train, test
import math

class ArgClass:
    def __init__(self, args):
        self.setattrs(**args)
        
    def setattrs(self, **kwargs):
        for name, val in kwargs.items():
            setattr(self, name, val)

class NetWithConv(nn.Module):
  def __init__(self, args, input_channels, image_size, num_labels):
    super().__init__()
    sparsities = getattr(args, "sparsity", [{"sparsity": 1.0}, {"sparsity": 1.0}, {"sparsity": 1.0}, {"sparsity": 1.0}, {"sparsity": 1.0}])
    self.conv1 = SupermaskConv(input_channels, 128, 3, 1, bias=False, sparsity=sparsities[0])
    self.conv2 = SupermaskConv(128, 128, 3, 1, bias=False, sparsity=sparsities[1])
    self.conv3 = SupermaskConv(128, 128, 3, 1, bias=False, sparsity=sparsities[1])
    # self.conv3 = SupermaskConv(64, 64, 3, 1, bias=False, sparsity=sparsities[1])
    # self.conv4 = SupermaskConv(64, 64, 3, 1, bias=False, sparsity=sparsities[1])
    # self.conv5 = SupermaskConv(64, 64, 3, 1, bias=False, sparsity=sparsities[1])
    #s = (image_size - 4) * (image_size - 4) * 64 // 4
    self.fc1 = SupermaskLinear(21632, 256, bias=False, sparsity=sparsities[2])
    self.fc2 = SupermaskLinear(256, 256, bias=False, sparsity=sparsities[3])
    self.fc3 = SupermaskLinear(256, num_labels, bias=False, sparsity=sparsities[4])
    self.args = args
  
  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = self.conv3(x)
    # x = F.relu(x)
    # x = self.conv3(x)
    # x = F.relu(x)
    # x = self.conv4(x)
    # x = F.relu(x)
    # x = self.conv5(x)
    x = F.max_pool2d(x, 2)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc3(x)
    output = F.log_softmax(x, dim=1)
    return output

  def forward_no_softmax(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = self.conv3(x)
    # x = F.relu(x)
    # x = self.conv4(x)
    # x = F.relu(x)
    # x = self.conv5(x)
    x = F.max_pool2d(x, 2)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc3(x)
    return x

  def get_pruned_weights(self):
      weights = []
      weights.append(self.conv1.pruned_weights())
      weights.append(self.conv2.pruned_weights())
      weights.append(self.conv3.pruned_weights())
      # weights.append(self.conv4.pruned_weights())
      # weights.append(self.conv5.pruned_weights())
      weights.append(self.fc1.pruned_weights())
      weights.append(self.fc2.pruned_weights())
      weights.append(self.fc3.pruned_weights())
      return weights
      
  def get_unpruned_weights(self):
    weights = []
    weights.append(self.conv1.weight)
    weights.append(self.conv2.weight)
    weights.append(self.conv3.weight)
    # weights.append(self.conv4.weight)
    # weights.append(self.conv5.weight)
    weights.append(self.fc1.weight)
    weights.append(self.fc2.weight)
    weights.append(self.fc3.weight)
    return weights

  def forward_unpruned(self, x):
    x = self.conv1.forward_unpruned(x)
    x = F.relu(x)
    x = self.conv2.forward_unpruned(x)
    x = F.relu(x)
    x = self.conv3.forward_unpruned(x)
    # x = F.relu(x)
    # x = self.conv4.forward_unpruned(x)
    # x = F.relu(x)
    # x = self.conv5.forward_unpruned(x)
    x = F.max_pool2d(x, 2)
    x = torch.flatten(x, 1)
    x = self.fc1.forward_unpruned(x)
    x = F.relu(x)
    x = self.fc2.forward_unpruned(x)
    x = F.relu(x)
    x = self.fc3.forward_unpruned(x)
    output = F.log_softmax(x, dim=1)
    return output

  def forward_unpruned_no_softmax(self, x):
    x = self.conv1.forward_unpruned(x)
    x = F.relu(x)
    x = self.conv2.forward_unpruned(x)
    x = F.relu(x)
    x = self.conv3.forward_unpruned(x)
    # x = F.relu(x)
    # x = self.conv4.forward_unpruned(x)
    # x = F.relu(x)
    # x = self.conv5.forward_unpruned(x)
    x = F.max_pool2d(x, 2)
    x = torch.flatten(x, 1)
    x = self.fc1.forward_unpruned(x)
    x = F.relu(x)
    x = self.fc2.forward_unpruned(x)
    x = F.relu(x)
    x = self.fc3.forward_unpruned(x)
    return x
  
  def get_extra_state(self):
      return self.args
    
  def set_extra_state(self, state):
      self.args = state

class Net(nn.Module):
    def __init__(self, args, input_size, num_layers, layers_size, output_size):
        super().__init__()
        sparsities = getattr(args, "sparsity", [1.0, 1.0, 1.0, 1.0])
        # self.conv1 = SupermaskConv(input_channels, 32, 3, 1, bias=False, sparsity=sparsities[0])
        # self.conv2 = SupermaskConv(32, 64, 3, 1, bias=False, sparsity=sparsities[1])
        # s = (image_size - 4) * (image_size - 4) * 64 // 4
        # self.fc1 = SupermaskLinear(s, 128, bias=False, sparsity=sparsities[2])
        # self.fc2 = SupermaskLinear(128, num_labels, bias=False, sparsity=sparsities[3])
        self.num_layers = num_layers
        self.linears = nn.ModuleList([SupermaskLinear(input_size, layers_size, bias=False, sparsity=sparsities[0])])
        self.linears.extend([SupermaskLinear(layers_size, layers_size, bias=False, sparsity = sparsities[i]) for i in range(1, self.num_layers-1)])
        self.linears.append(SupermaskLinear(layers_size, output_size, bias=False, sparsity = sparsities[self.num_layers - 1]))

        self.args = args

    def forward(self, x):
        for i in range(self.num_layers - 1):
          x = self.linears[i](x)
          x = F.tanh(x)
        x = self.linears[-1](x)
        output = F.log_softmax(x, dim=1)
        return output

    def forward_no_softmax(self, x):
      for i in range(self.num_layers - 1):
          x = self.linears[i](x)
          x = F.tanh(x)
      x = self.linears[-1](x)
      return x

    def get_pruned_weights(self):
      return [linear.pruned_weights().T for linear in self.linears]
    
    def get_unpruned_weights(self):
      return [linear.weight.T for linear in self.linears]

    def forward_unpruned(self, x):
      for i in range(self.num_layers - 1):
          x = self.linears[i].forward_unpruned(x)
          x = F.tanh(x)
      x = self.linears[-1].forward_unpruned(x)
      output = F.log_softmax(x, dim=1)
      return output

    def forward_unpruned_no_softmax(self, x):
      for i in range(self.num_layers - 1):
          x = self.linears[i].forward_unpruned(x)
          x = F.tanh(x)
      x = self.linears[-1].forward_unpruned(x)
      return x
    
    def get_extra_state(self):
        return self.args
      
    def set_extra_state(self, state):
        self.args = state


def prune_and_train(model_args, prune_args, train_args, base_model=None, cnn=False):
  trained_model, device, train_loader, test_loader, criterion, prune_acc, prune_loss = main(model_args, prune_args, cnn=cnn)
  pruned_model = copy.deepcopy(trained_model)
  trained_model, device, train_loader, test_loader, criterion, train_acc, train_loss = train_after(trained_model, train_args, device, train_loader, test_loader)
  return trained_model, pruned_model, device, train_loader, test_loader, criterion, np.array(prune_acc), np.array(prune_loss), np.array(train_acc), np.array(train_loss)


# The main function runs the full training loop on a dataset of your choice
def main(model_args, train_args, cnn=False, base_model=None):
    args = ArgClass(model_args)
    train_args = ArgClass(train_args)
    dataset = args.dataset
    accuracy = []
    losses = []

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device {device}")

    transform = None
    if dataset == "MNIST" and cnn:
        transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])
        input_channels, image_size, num_labels = 1, 28, 10
    elif dataset == "CIFAR10" and cnn:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                              ])
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                        ])
        input_channels, image_size, num_labels = 3, 32, 10
    elif dataset == "MNIST" and not cnn:
        transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.1307,), (0.3081,)),
                                        transforms.Lambda(lambda x: torch.flatten(x))
                                        ])
        input_channels, image_size, num_labels = 1, 28, 10
    elif dataset == "CIFAR10" and not cnn:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        transforms.Lambda(lambda x: torch.flatten(x))
                                        ])
        input_channels, image_size, num_labels = 3, 32, 10
    else:
        raise ValueError("Only supported datasets are CIFAR10 and MNIST currently.")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        getattr(datasets, dataset)(os.path.join(train_args.data, dataset), 
                                   train=True, download=True, transform=train_transform),
        batch_size=args.batch_size, shuffle=True,**kwargs)
    test_loader = torch.utils.data.DataLoader(
        getattr(datasets, dataset)(os.path.join(train_args.data, dataset), 
                                   train=False, transform=transform),
        batch_size=train_args.test_batch_size, shuffle=True, **kwargs)
    if not cnn:
      model = Net(args, input_size, num_layers, layer_size,10).to(device)
    else:
      model = NetWithConv(args, 3, 32, 10).to(device)
    # if getattr(args, "copy_layers", None) is not None:
    #     if (bool(args.copy_layers) ^ (base_model is not None)):
    #         raise ValueError("copy_layers arg must be None or [] if base_model is not specified")
    #     if base_model is not None and args.copy_layers:
    #         for layer in args.copy_layers:
    #             model.load_state_dict(getattr(base_model, layer).state_dict(prefix=f"{layer}."), strict=False)
                
    # if getattr(args, "freeze_layers", None):
    #     for layer_name in args.freeze_layers:
    #         getattr(model, layer_name).freeze()
            
    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )
    assert isinstance(args.epochs, list) or isinstance(args.epochs, int)
    num_epochs, check_freeze = (args.epochs, False) if isinstance(args.epochs, int) else (max(args.epochs), True)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    for epoch in range(1, num_epochs + 1):
        # if check_freeze:
        #     for freeze_at_epoch, child in zip(args.epochs, model.children()):
        #         if freeze_at_epoch == epoch - 1:
        #             child.freeze()
        #             print(f"Freezing {child} before epoch {epoch}")
        train(model, train_args.log_interval, device, train_loader, optimizer, criterion, epoch, alpha=train_args.alpha, loss=train_args.reg)
        if (train_args.train_eval_interval and epoch % train_args.train_eval_interval == 0) or (train_args.eval_train_on_last and epoch == args.epochs):
            test(model, device, criterion, train_loader, name="Train", alpha=train_args.alpha, loss=train_args.reg)
        if (train_args.test_eval_interval and epoch % train_args.test_eval_interval == 0) or (train_args.eval_test_on_last and epoch == args.epochs):
            acc, loss = test(model, device, criterion, test_loader, name="Test", alpha=train_args.alpha, loss=train_args.reg)
            accuracy.append(acc)
            losses.append(loss)
        scheduler.step()

    # if args.save_name is not None:
    #     torch.save(model.state_dict(), os.path.join(os.environ['HOME_DIR'], \
    #                                                 "trained_networks", args.save_name))
    
    return model, device, train_loader, test_loader, criterion, accuracy, losses

def train_after(trained_model, args, device, train_loader, test_loader):
  args = ArgClass(args)
  accuracy = []
  losses = []
  for name, l in trained_model.named_parameters():
    if 'weight' in name:
      l.requires_grad = True
    else:
      l.requires_grad = False

  optimizer = optim.SGD(
          [p for p in trained_model.parameters() if p.requires_grad],
          lr=args.lr,
          momentum=args.momentum,
          weight_decay=args.wd,
      )
  assert isinstance(args.epochs, list) or isinstance(args.epochs, int)
  num_epochs, check_freeze = (args.epochs, False) if isinstance(args.epochs, int) else (max(args.epochs), True)
  criterion = nn.CrossEntropyLoss().to(device)
  for epoch in range(1, num_epochs + 1):
      if check_freeze:
          for freeze_at_epoch, child in zip(args.epochs, trained_model.children()):
              if freeze_at_epoch == epoch - 1:
                  child.freeze()
                  print(f"Freezing {child} before epoch {epoch}")

      train(trained_model, args.log_interval, device, train_loader, optimizer, criterion, epoch,loss=args.reg)
      if (args.train_eval_interval and epoch % args.train_eval_interval == 0) or (args.eval_train_on_last and epoch == args.epochs):
          test(trained_model, device, criterion, train_loader, name="Train",loss=args.reg)
      if (args.test_eval_interval and epoch % args.test_eval_interval == 0) or (args.eval_test_on_last and epoch == args.epochs):
          acc, loss = test(trained_model, device, criterion, test_loader, name="Test",loss=args.reg)
          accuracy.append(acc)
          losses.append(loss)
  return trained_model, device, train_loader, test_loader, criterion, accuracy, losses

def get_prune_mask(layer, sparsity):
    with torch.no_grad():
        return GetSubnet.apply(layer.scores.abs(), sparsity)

def find_var(model, point, linearity, pruned=False):
  output = []
  vars = []
  l = model.linears
  if pruned:
    output.append(l[0](point))
  else:
    output.append(l[0].forward_unpruned(point))
  for i in range(1, len(l)):
    if i != len(l) - 1:
      if pruned:
        output.append(l[i](linearity(output[i-1])))
      else:
        output.append(l[i].forward_unpruned(linearity(output[i-1])))
    else:
      if pruned:
        output.append(l[i](F.log_softmax(output[i-1])))
      else:
        output.append(l[i].forward_unpruned(F.log_softmax(output[i-1])))
  vars.append(torch.square(torch.norm(point)).item() / l[0].in_features)
  for idx, out in enumerate(output):
    vars.append(torch.square(torch.norm(out)).item() / l[idx].out_features)

  return output , vars

def compute_jacobian(f, x, prune=True):
    '''
    Normal:
        f: input_dims -> output_dims
    Jacobian mode:
        f: output_dims x input_dims -> output_dims x output_dims
    '''
    if prune:
      return jacobian(f, x.to('cuda'), vectorize=True)
    else:
      return jacobian(f.forward_unpruned, x.to('cuda'), vectorize=True)
      

def get_singular_values(model, point, prune=True):
  J = compute_jacobian(model, point, prune=prune)
  return torch.linalg.svdvals(J).squeeze()

def find_covar(model, point1, point2, linearity, pruned=False):
  output1 = []
  output2 = []
  covars = []
  l = model.linears
  if pruned:
    output1.append(l[0](point1))
    output2.append(l[0](point2))
  else:
    output1.append(l[0].forward_unpruned(point1))
    output2.append(l[0].forward_unpruned(point2))
  for i in range(1, len(l)):
    if i != len(l) - 1:
      if pruned:
        output1.append(l[i](linearity(output1[i-1])))
        output2.append(l[i](linearity(output2[i-1])))
      else:
        output1.append(l[i].forward_unpruned(linearity(output1[i-1])))
        output2.append(l[i].forward_unpruned(linearity(output2[i-1])))
    else:
      if pruned:
        output1.append(l[i](F.log_softmax(output1[i-1])))
        output2.append(l[i](F.log_softmax(output2[i-1])))
      else:
        output1.append(l[i].forward_unpruned(F.log_softmax(output1[i-1])))
        output2.append(l[i].forward_unpruned(F.log_softmax(output2[i-1])))
  covars.append(torch.dot(point1.squeeze(0), point2.squeeze(0)).item() / l[0].in_features)
  for idx, out in enumerate(output1):
    covars.append(torch.dot(out.squeeze(0), output2[idx].squeeze(0)).item() / l[idx].out_features)

  return output1, output2 , covars

def plot_confidence(total_var, start, end):
  mean_var = np.mean(total_var, axis= 1)
  fifth = np.percentile(total_var, 5, axis=1)
  ninty_fifth = np.percentile(total_var, 95, axis=1)

  layer = np.arange(start, end)
  plt.plot(layer, mean_var[start:-1])
  # plt.plot(layer, pruned_fifth[1:-1])
  # plt.plot(layer, pruned_ninty_fifth[1:-1])
  plt.fill_between(layer, fifth[start:-1], ninty_fifth[start:-1], color='b', alpha=.2)
  plt.xlabel('Layer Number')
  plt.ylabel('Variance at each layer')
  plt.show()

def get_condition_number(model, point, prune=True):
  sv = get_singular_values(model, point, prune)
  return max(sv) / min(sv)

