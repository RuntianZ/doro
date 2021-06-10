"""Improving CelebA in torchvision"""
import torchvision
import os

from dataset.utils import download_and_merge, download

class CelebA(torchvision.datasets.CelebA):
  def download(self):
    import zipfile

    if self._check_integrity():
      print('Files already downloaded and verified')
      return

    print('Downloading CelebA...')
    fpath = os.path.join(self.root, self.base_folder, "img_align_celeba.zip")
    download_and_merge("https://raw.githubusercontent.com/RuntianZ/CelebaFiles/master/img_align_celeba.zip",
                       fpath, 73)

    fpath = os.path.join(self.root, self.base_folder, "list_attr_celeba.txt")
    download_and_merge("https://raw.githubusercontent.com/RuntianZ/CelebaFiles/master/list_attr_celeba.txt",
                       fpath, 2)

    for (_, md5, filename) in self.file_list:
      if filename in ["img_align_celeba.zip", "list_attr_celeba.txt"]:
        continue
      fpath = os.path.join(self.root, self.base_folder, filename)
      url = "https://raw.githubusercontent.com/RuntianZ/CelebaFiles/master/{}".format(filename)
      download(url, fpath)

    with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
      f.extractall(os.path.join(self.root, self.base_folder))

    if not self._check_integrity():
      raise RuntimeError('Failed to download CelebA')
    print('Download complete.')