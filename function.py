import cv2
import torch
import os
import utils
from config import Config
from torch.nn import functional as F
import collections
import numpy as np
import random


def worker_init_fn(worker_id):
  worker_seed = torch.initial_seed() % 2 ** 32
  np.random.seed(worker_seed)
  random.seed(worker_seed)


def load_pre_weights(model, filepath):
  if os.path.exists(filepath):
    pretrained_dict = torch.load(filepath)
    model_dict = model.state_dict()
    if "resnet" in filepath:
      filtered_dict = collections.OrderedDict()
      del pretrained_dict['fc.weight']
      del pretrained_dict['fc.bias']

      for k, v in pretrained_dict.items():
        if k.startswith("layer3") or k.startswith("layer4"):
          layersl = k[0:6] + "l" + k[6:]
          layersh = k[0:6] + "h" + k[6:]
          filtered_dict["module." + layersl] = v
          filtered_dict["module." + layersh] = v
        else:
          filtered_dict["module." + k] = v
        pretrained_dict = filtered_dict

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
  else:
    print("Weight file not found ...")


def inference(model, image):
  model.eval()
  with torch.no_grad():
    image_resize = cv2.resize(image, (Config.base_size, Config.base_size))
    image_resize = utils.image_to_tensor(image_resize, Config.mean, Config.std)

    mask_pred, edge_pred = model(image_resize)
    mask_pred = F.interpolate(input=mask_pred, size=image.shape[:2], mode='bilinear',
                              align_corners=Config.align_corners)
    mask_pred = torch.sigmoid(mask_pred)
    mask_pred = (mask_pred.squeeze(0).squeeze(0) >= 0.5).long()
    return mask_pred.cpu().numpy()










