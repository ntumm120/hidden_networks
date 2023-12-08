import math
import torch

def ortho_reg(weights, gain = math.sqrt(2)):
  reg = 0
  for W in weights:
    if W.ndimension() < 2:
      continue
    else:
      cols = W[0].numel()
      rows = W.shape[0]
      w1 = W.view(-1,cols)
      if w1.shape[0] < w1.shape[1]:
        w1 = w1.permute(1,0)
      wt = torch.transpose(w1,0,1)
      m  = torch.matmul(wt,w1)
      reg += torch.norm((m - gain ** 2 * torch.eye(w1.shape[1]).cuda()))
  return reg

def srip_reg(weights, gain=math.sqrt(2)):
  srip_reg = 0
  for W in weights:
    if W.ndimension() < 2:
        continue
    else:
      cols = W[0].numel()
      rows = W.shape[0]
      w1 = W.view(-1,cols)
      if w1.shape[0] < w1.shape[1]:
        w1 = w1.permute(1,0)
      wt = torch.transpose(w1,0,1)
      m  = torch.matmul(wt,w1)
      w_tmp = m - gain ** 2 * torch.eye(w1.shape[1]).cuda()
      height = w_tmp.size(0)
      u = torch.nn.functional.normalize(w_tmp.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
      v = torch.nn.functional.normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
      u = torch.nn.functional.normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
      sigma = torch.dot(u, torch.matmul(w_tmp, v))
      srip_reg = srip_reg + (torch.norm(sigma,2))**2
  return srip_reg