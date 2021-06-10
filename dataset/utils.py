import os
from six.moves import urllib
import tempfile
from urllib.error import HTTPError, URLError
import sys
import time
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def download(url, localpath, verbose=False):
  """Robust download"""
  if not os.path.isdir(os.path.dirname(os.path.abspath(localpath))):
    os.makedirs(os.path.dirname(os.path.abspath(localpath)))
  if verbose:
    print('Downloading %s to %s' % (url, localpath))
  for i in range(20):
    try:
      urllib.request.urlretrieve(url, localpath)
      return
    except (HTTPError, URLError, ConnectionResetError) as e:
      print(e)
      sys.stderr.write('Network error. Retry {}\n'.format(i + 1))
      time.sleep(5)
  raise RuntimeError('Network error occurs 20 times')


def merge_file(fromfile, tofile):
  if not os.path.isdir(os.path.dirname(os.path.abspath(tofile))):
    os.makedirs(os.path.dirname(os.path.abspath(tofile)))
  partnum = 1
  with open(tofile, 'wb') as fout:
    while True:
      fname = '{}.part{}'.format(fromfile, partnum)
      if not os.path.exists(fname):
        break
      with open(fname, 'rb') as fin:
        chunk = fin.read()
        fout.write(chunk)
      partnum += 1
    partnum -= 1
  return partnum


def download_and_merge(url, tofile, part_num):
  if not os.path.isdir(os.path.dirname(os.path.abspath(tofile))):
    os.makedirs(os.path.dirname(os.path.abspath(tofile)))
  if part_num == 0:
    download(url, tofile)
  else:
    tmp = tempfile.TemporaryDirectory()
    with open(tofile, 'wb') as fout:
      for i in range(part_num):
        file_url = '{}.part{}'.format(url, i + 1)
        local_pth = '{}/tmp.part{}'.format(tmp.name, i + 1)
        download(file_url, local_pth)
        with open(local_pth, 'rb') as fin:
          chunk = fin.read()
          fout.write(chunk)
        os.remove(local_pth)
