"""Experiments on CelebA

Algorithms (--alg): erm, cvar, cvar_doro, chisq, chisq_doro
Use --download to download the dataset if you are running for the first time.
"""

from dro import *

import os
import argparse
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from dataset.celeba import CelebA
from torchvision.models import resnet18
import torchvision.transforms as transforms


def get_transform_celebA(augment, target_w=None, target_h=None):
  # Reference: https://github.com/kohpangwei/group_DRO/blob/f7eae929bf4f9b3c381fae6b1b53ab4c6c911a0e/data/celebA_dataset.py#L80
  orig_w = 178
  orig_h = 218
  orig_min_dim = min(orig_w, orig_h)
  if target_w is not None and target_h is not None:
    target_resolution = (target_w, target_h)
  else:
    target_resolution = (orig_w, orig_h)

  if not augment:
    transform = transforms.Compose([
      transforms.CenterCrop(orig_min_dim),
      transforms.Resize(target_resolution),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  else:
    # Orig aspect ratio is 0.81, so we don't squish it in that direction any more
    transform = transforms.Compose([
      transforms.RandomResizedCrop(
        target_resolution,
        scale=(0.7, 1.0),
        ratio=(1.0, 1.3333333333333333),
        interpolation=2),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  return transform


def main():
  parser = argparse.ArgumentParser()

  # Basic settings
  parser.add_argument('--data_root', type=str)
  parser.add_argument('--device', default='cuda', type=str)
  parser.add_argument('--seed', type=int)
  parser.add_argument('--save_file', type=str)
  parser.add_argument('--download', default=False, action='store_true')

  # Training settings
  parser.add_argument('--alg', type=str)
  parser.add_argument('--epochs', default=30, type=int)
  parser.add_argument('--batch_size', default=400, type=int)
  parser.add_argument('--lr', default=0.001, type=float)
  parser.add_argument('--wd', default=0.001, type=float)
  parser.add_argument('--scheduler', type=str)
  parser.add_argument('--alpha', type=float)
  parser.add_argument('--eps', type=float)

  args = parser.parse_args()
  print('Algorithm: {}'.format(args.alg))
  print('alpha: {}'.format(args.alpha))
  print('eps: {}'.format(args.eps))
  print('Batch size: {}'.format(args.batch_size))
  print('lr: {}'.format(args.lr))
  print('wd: {}'.format(args.wd))
  print('Epochs: {}'.format(args.epochs))

  data_root = args.data_root
  device = args.device
  if args.save_file is not None:
    d = os.path.dirname(os.path.abspath(args.save_file))
    if not os.path.isdir(d):
      os.makedirs(d)

  # Prepare dataset
  target_w = 224
  target_h = 224
  n_classes = 2
  transform_train = get_transform_celebA(True, target_w, target_h)
  transform_test = get_transform_celebA(False, target_w, target_h)


  dataset_test = CelebA(data_root, split='test', target_type='attr',
                        transform=transform_test, download=args.download)
  dataset_valid = CelebA(data_root, split='valid', target_type='attr',
                         transform=transform_test, download=False)
  target_idx = 9  # Blond

  # Domains
  domain_fn = [
    lambda t: (t[:, 20] == 1) & (t[:, 9] == 1),  # Male            Blond
    lambda t: (t[:, 20] == 1) & (t[:, 9] == 0),  # Male            Not-Blond
    lambda t: (t[:, 20] == 0) & (t[:, 9] == 1),  # Female          Blond
    lambda t: (t[:, 20] == 0) & (t[:, 9] == 0),  # Female          Not-Blond
    lambda t: (t[:, 39] == 1) & (t[:, 9] == 1),  # Young           Blond
    lambda t: (t[:, 39] == 1) & (t[:, 9] == 0),  # Young           Not-Blond
    lambda t: (t[:, 39] == 0) & (t[:, 9] == 1),  # Old             Blond
    lambda t: (t[:, 39] == 0) & (t[:, 9] == 0),  # Old             Not-Blond
    lambda t: (t[:, 2] == 1) & (t[:, 9] == 1),   # Attractive      Blond
    lambda t: (t[:, 2] == 1) & (t[:, 9] == 0),   # Attractive      Not-Blond
    lambda t: (t[:, 2] == 0) & (t[:, 9] == 1),   # Not-Attractive  Blond
    lambda t: (t[:, 2] == 0) & (t[:, 9] == 0),   # Not-Attractive  Not-Blond
    lambda t: (t[:, 32] == 1) & (t[:, 9] == 1),  # Straight-Hair   Blond
    lambda t: (t[:, 32] == 1) & (t[:, 9] == 0),  # Straight-Hair   Not-Blond
    lambda t: (t[:, 33] == 1) & (t[:, 9] == 1),  # Wavy-Hair       Blond
    lambda t: (t[:, 33] == 1) & (t[:, 9] == 0),  # Wavy-Hair       Not-Blond
  ]

  label_id = lambda t: t[:, target_idx]
  dataset_train = CelebA(data_root, split='train', target_type='attr',
                         transform=transform_train,
                         target_transform=lambda t: t[target_idx])

  # Fix seed for reproducibility
  if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_deterministic(True)
    cudnn.benchmark = False
  else:
    cudnn.benchmark = True

  # Build model
  model = resnet18()
  d = model.fc.in_features
  model.fc = nn.Linear(d, n_classes)
  model = model.to(device)
  model = torch.nn.DataParallel(model)

  trainloader = DataLoader(dataset_train, batch_size=args.batch_size,
                           num_workers=4, pin_memory=True)
  testloader = DataLoader(dataset_test, batch_size=args.batch_size,
                          num_workers=4, pin_memory=True)
  validloader = DataLoader(dataset_valid, batch_size=args.batch_size,
                           num_workers=4, pin_memory=True)
  optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.wd)
  criterion = nn.CrossEntropyLoss(reduction='none')
  scheduler = None
  if args.scheduler is not None:
    milestones = args.scheduler.split(',')
    milestones = [int(s) for s in milestones]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

  # Training
  val_avg_acc = []
  val_avg_loss = []
  val_group_acc = []
  val_group_loss = []
  val_cvar_loss = []
  val_cvar_doro_loss = []
  avg_acc = []
  avg_loss = []
  group_acc = []
  group_loss = []
  best_valid = 0.
  best_epoch = 0
  best_acc = 0.
  best_worst_acc = 0.
  for epoch in range(args.epochs):
    print('===Train(epoch={})==='.format(epoch + 1))
    train(args.alg, model, trainloader, optimizer, criterion, device,
          args.alpha, args.eps)
    if scheduler is not None:
      scheduler.step()
    print('===Validation(epoch={})==='.format(epoch + 1))
    a, b, c, d, e, f = test(model, validloader, criterion, device, domain_fn, label_id, True)
    val_avg_acc.append(a)
    val_avg_loss.append(b)
    val_group_acc.append(c)
    val_group_loss.append(d)
    val_cvar_loss.append(e)
    val_cvar_doro_loss.append(f)
    worst_acc = c.min()
    if worst_acc > best_valid:
      best_valid = worst_acc
      best_epoch = epoch + 1
    print('===Test(epoch={})==='.format(epoch + 1))
    a, b, c, d = test(model, testloader, criterion, device, domain_fn, label_id)
    worst_acc = c.min()
    if best_epoch == epoch + 1:
      best_acc = a
      best_worst_acc = worst_acc
    avg_acc.append(a)
    avg_loss.append(b)
    group_acc.append(c)
    group_loss.append(d)

  # Print the results
  print('===Results===')
  print('                           Best Epoch: {}'.format(best_epoch))
  print('      Test Accuracy of the Best Epoch: {}'.format(best_acc))
  print('Worst-case Accuracy of the Best Epoch: {}'.format(best_worst_acc))

  # Save the results
  if args.save_file is not None:
    mat = {
      'avg_acc': np.array(avg_acc),
      'avg_loss': np.array(avg_loss),
      'group_acc': np.array(group_acc),
      'group_loss': np.array(group_loss),
      'val_avg_acc': np.array(val_avg_acc),
      'val_avg_loss': np.array(val_avg_loss),
      'val_group_acc': np.array(val_group_acc),
      'val_group_loss': np.array(val_group_loss),
      'val_cvar_loss': np.array(val_cvar_loss),
      'val_cvar_doro_loss': np.array(val_cvar_doro_loss),
      'best_epoch': best_epoch,
      'best_acc': best_acc,
      'best_worst_acc': best_worst_acc,
    }
    sio.savemat(args.save_file, mat)


def test(model: Module, loader: DataLoader, criterion, device: str,
         domain_fn, label_id, need_cvar=False):
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
  alpha = 0.2
  eps = 0.005

  with torch.no_grad():
    for _, (inputs, targets) in enumerate(loader):
      inputs, targets = inputs.to(device), targets.to(device)

      labels = label_id(targets)
      outputs = model(inputs)
      predictions = torch.argmax(outputs, dim=1)
      c = (predictions == labels)
      correct = c.sum().item()
      l = criterion(outputs, labels).view(-1)
      if need_cvar:
        l_rec.append(l.detach().cpu().numpy())
      loss = l.sum().item()
      total_correct += correct
      total_loss += loss
      total_num += len(inputs)

      for i in range(num_domains):
        g = domain_fn[i](targets)
        group_correct[i] += c[g].sum().item()
        group_loss[i] += l[g].sum().item()
        group_num[i] += g.sum().item()

  print('Acc: {} ({} of {})'.format(total_correct / total_num, total_correct, total_num))
  print('Avg Loss: {}'.format(total_loss / total_num))
  for i in range(num_domains):
    print('Group {:2}\tAcc: {} ({} of {})'.format(i, group_correct[i] / group_num[i],
                                                group_correct[i], group_num[i]))
    print('Group {:2}\tAvg Loss: {}'.format(i, group_loss[i] / group_num[i]))

  if need_cvar:
    l_vec = np.concatenate(l_rec)
    n = int(len(l_vec) * alpha)
    l = np.sort(l_vec)
    l1 = l[-n:]
    cvar_loss = l1.mean()
    print('CVaR loss: {}'.format(cvar_loss))

    n1 = int(len(l_vec) * (eps + alpha * (1 - eps)))
    n2 = int(len(l_vec) * eps)
    l2 = l[-n1:-n2]
    cvar_doro_loss = l2.mean()
    print('CVaR-DORO loss: {}'.format(cvar_doro_loss))
    return total_correct / total_num, total_loss / total_num, \
           group_correct / group_num, group_loss / group_num, \
           cvar_loss, cvar_doro_loss

  return total_correct / total_num, total_loss / total_num, \
         group_correct / group_num, group_loss / group_num


if __name__ == '__main__':
  main()
