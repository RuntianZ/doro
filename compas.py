"""Experiments on COMPAS

Algorithms (--alg): erm, cvar, cvar_doro, chisq, chisq_doro
Use --data_mat to specify a user training set (.mat file)
Use --remove_outliers to remove outliers from the dataset
Use --noise to add noise to the dataset
"""

from dro import *

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as sio

import torch
import torch.nn
from torch import Tensor
from torch import optim
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


def build_model(input_dim: int) -> Module:
  """Two-layer feed-forward ReLU neural network"""

  model = torch.nn.Sequential(torch.nn.Linear(input_dim, 10, bias=True),
                              torch.nn.ReLU(inplace=True),
                              torch.nn.Linear(10, 1, bias=True))
  return model


def preprocess_compas(df: pd.DataFrame):
  """Preprocess dataset"""

  columns = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
             'age',
             'c_charge_degree',
             'sex', 'race', 'is_recid']
  target_variable = 'is_recid'

  df = df[['id'] + columns].drop_duplicates()
  df = df[columns]

  race_dict = {'African-American': 1, 'Caucasian': 0}
  df['race'] = df.apply(lambda x: race_dict[x['race']] if x['race'] in race_dict.keys() else 2, axis=1).astype(
    'category')

  sex_map = {'Female': 0, 'Male': 1}
  df['sex'] = df['sex'].map(sex_map)

  c_charge_degree_map = {'F': 0, 'M': 1}
  df['c_charge_degree'] = df['c_charge_degree'].map(c_charge_degree_map)

  X = df.drop([target_variable], axis=1)
  y = df[target_variable]
  return X, y


class MyDataset(Dataset):
  def __init__(self, X, y):
    super(MyDataset, self).__init__()
    self.X = X
    self.y = y

  def __getitem__(self, item):
    return self.X[item], self.y[item]

  def __len__(self):
    return len(self.X)
  

class MyLoss(object):
  def __init__(self, reduction='mean'):
    self.reduction = reduction

  def __call__(self, outputs: Tensor, targets: Tensor) -> Tensor:
    outputs = outputs.view(-1)
    loss = -targets * F.logsigmoid(outputs) - (1 - targets) * F.logsigmoid(-outputs)
    if self.reduction == 'mean':
      loss = loss.mean()
    elif self.reduction == 'sum':
      loss = loss.sum()
    return loss


def main():
  parser = argparse.ArgumentParser()
  
  # Dataset args
  parser.add_argument('--data_mat', type=str)
  parser.add_argument('--noise', default=0.0, type=float)

  # Remove outliers: this only output a clean dataset but does not train on it
  parser.add_argument('--remove_outliers', default=False, action='store_true')
  parser.add_argument('--trim_times', default=5, type=int)
  parser.add_argument('--trim_num', default=200, type=int)

  # Train args
  parser.add_argument('--alg', default='erm', type=str)
  parser.add_argument('--batch_size', default=128, type=int)
  parser.add_argument('--device', default='cpu', type=str)
  parser.add_argument('--lr', default=0.01, type=float)
  parser.add_argument('--alpha', type=float)
  parser.add_argument('--eps', type=float)
  parser.add_argument('--epochs', default=300, type=int)

  parser.add_argument('--save_file', type=str)
  parser.add_argument('--seed', type=int)
  args = parser.parse_args()
  device = args.device
  if args.save_file is not None:
    d = os.path.dirname(os.path.abspath(args.save_file))
    if not os.path.isdir(d):
      os.makedirs(d)

  # Prepare the dataset
  df = pd.read_csv('compas-scores-two-years.csv')
  X, y = preprocess_compas(df)
  input_dim = len(X.columns)
  X, y = X.to_numpy().astype('float32'), y.to_numpy()
  X[:, 4] /= 10
  X[X[:, 7] > 0, 7] = 1 # Race: White (0) and Others (1)
  domain_fn = [
    lambda x: x[:, 7] == 0, # White
    lambda x: x[:, 7] == 1, # Others
    lambda x: x[:, 6] == 0, # Female
    lambda x: x[:, 6] == 1, # Male
  ]

  # Split the dataset: train-test = 70-30
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                      random_state=42, shuffle=True)

  if args.data_mat is not None:
    # User dataset
    mat = sio.loadmat(args.data_mat)
    X_train = mat['X_train'].astype('float32')
    y_train = mat['y_train'].flatten()

  trainset = MyDataset(X_train, y_train)
  testset = MyDataset(X_test, y_test)
  trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
  testtrainloader = DataLoader(trainset, batch_size=1024, shuffle=False)
  testloader = DataLoader(testset, batch_size=1024, shuffle=False)

  if args.seed is not None:
    # Fix seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_deterministic(True)
    cudnn.benchmark = False
  else:
    cudnn.benchmark = True

  # Label noise
  if args.noise > 0.0:
    l = len(y_train)
    n_noise = int(l * args.noise)
    r = np.random.rand(l)
    r = np.argsort(r)
    y_train[r[:n_noise]] = 1 - y_train[r[:n_noise]]

  # Remove outliers
  if args.remove_outliers:
    assert args.save_file is not None
    for t in range(args.trim_times):
      print('=====Round {}====='.format(t + 1))
      model = build_model(input_dim)
      model = model.to(device)
      optimizer = optim.ASGD(model.parameters(), lr=0.1)
      criterion = MyLoss(reduction='none')
      for epoch in range(args.epochs):
        print('Epoch {}'.format(epoch + 1))
        train(args.alg, model, trainloader, optimizer, criterion, device,
              args.alpha, args.eps)
        test(model, testloader, criterion, device, domain_fn)

      # Trim
      r = test(model, testtrainloader, criterion, device,
               domain_fn, args.trim_num)
      X_train = X_train[r]
      y_train = y_train[r]
      trainset = MyDataset(X_train, y_train)
      trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
      testtrainloader = DataLoader(trainset, batch_size=1024, shuffle=False)

    mat = {
      'X_train': X_train,
      'y_train': y_train,
    }
    sio.savemat(args.save_file, mat)
    return

  # Training
  model = build_model(input_dim)
  model = model.to(device)
  criterion = MyLoss(reduction='none')
  optimizer = optim.ASGD(model.parameters(), lr=args.lr)

  avg_acc = []
  avg_loss = []
  group_acc = []
  group_loss = []
  for epoch in range(args.epochs):
    print('Epoch {}'.format(epoch + 1))
    train(args.alg, model, trainloader, optimizer, criterion, device,
          args.alpha, args.eps)
    a, b, c, d = test(model, testloader, criterion, device, domain_fn)
    avg_acc.append(a)
    avg_loss.append(b)
    group_acc.append(c)
    group_loss.append(d)

  # Save the results
  if args.save_file is not None:
    mat = {
      'avg_acc': np.array(avg_acc),
      'avg_loss': np.array(avg_loss),
      'group_acc': np.array(group_acc),
      'group_loss': np.array(group_loss),
    }
    sio.savemat(args.save_file, mat)

  
def test(model: Module, loader: DataLoader, criterion, device: str,
         domain_fn, trim_num=None):
  """Test the avg and group acc of the model"""
  
  model.eval()
  total_correct = 0
  total_loss = 0
  total_num = 0
  num_domains = len(domain_fn)
  group_correct = np.zeros((num_domains,), dtype=np.int)
  group_loss = np.zeros((num_domains,), dtype=np.float)
  group_num = np.zeros((num_domains,), dtype=np.int)
  l_rec = []
  
  with torch.no_grad():
    for _, (inputs, targets) in enumerate(loader):
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs).view(-1)
      c = ((outputs > 0) & (targets == 1)) | ((outputs < 0) & (targets == 0))
      correct = c.sum().item()
      l = criterion(outputs, targets).view(-1)
      if trim_num is not None:
        l_rec.append(l.detach().cpu().numpy())
      loss = l.sum().item()
      total_correct += correct
      total_loss += loss
      total_num += len(inputs)

      for i in range(num_domains):
        g = domain_fn[i](inputs)
        group_correct[i] += c[g].sum().item()
        group_loss[i] += l[g].sum().item()
        group_num[i] += g.sum().item()
        
  print('Acc: {} ({} of {})'.format(total_correct / total_num, total_correct, total_num))
  print('Avg Loss: {}'.format(total_loss / total_num))
  for i in range(num_domains):
    print('Group {}\tAcc: {} ({} of {})'.format(i, group_correct[i]/ group_num[i],
                                               group_correct[i], group_num[i]))
    print('Group {}\tAvg Loss: {}'.format(i, group_loss[i] / group_num[i]))

  if trim_num is not None:
    l_vec = np.concatenate(l_rec)
    l = np.argsort(l_vec)[:-trim_num]
    return l

  return total_correct / total_num, total_loss / total_num, \
         group_correct / group_num, group_loss / group_num


if __name__ == '__main__':
  main()
  