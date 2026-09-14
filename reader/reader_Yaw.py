import numpy as np
import cv2 
import os
from torch.utils.data import Dataset, DataLoader
import torch
from easydict import EasyDict as edict

def gazeto2d(gaze):
  yaw = np.arctan2(-gaze[0], -gaze[2])
  pitch = np.arcsin(-gaze[1])
  return np.array([yaw, pitch])

class loader(Dataset): 
  def __init__(self, path, root, header=True):
    self.lines = []
    if isinstance(path, list):
      for i in path:
        with open(i) as f:
          line = f.readlines()
          if header: line.pop(0)
          self.lines.extend(line)
    else:
      with open(path) as f:
        self.lines = f.readlines()
        if header: self.lines.pop(0)

    self.root = root

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.replace("\\", "/")
    line = line.strip().split(" ")

    face = line[0]
    name = face.replace('/', '_')
    paired_face = line[1]
    
    fimg = cv2.imread(os.path.join(self.root, face))/255.0
    fimg = cv2.resize(fimg, (224, 224))
    fimg = fimg.transpose(2, 0, 1)
    
    pfimg = cv2.imread(os.path.join(self.root, paired_face))/255.0
    pfimg = cv2.resize(pfimg, (224, 224))
    fipfimgmg = pfimg.transpose(2, 0, 1)
    
    label = 0
    pairedlabel = 0
    data = edict()
    data.face = fimg
    data.name = name
    data.pairedface = fipfimgmg

    return data, label, pairedlabel

def txtload(labelpath, imagepath, batch_size, shuffle=True, num_workers=0, header=True):
  dataset = loader(labelpath, imagepath, header)
  print(f"[Read Data]: Total num: {len(dataset)}")
  print(f"[Read Data]: Label path: {labelpath}")
  load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  return load


if __name__ == "__main__":
  path = '/home/huds/YawDDFace/YawDD.txt'
  root = '/home/huds/YawDDFace'
  d = loader(path, root)
  print(len(d))
  (data, label, plabel) = d.__getitem__(0)
  print(data)

