import math
import torch

def input_transform(image, mean, std):
  image = image / 255.0
  image -= mean
  image /= std
  return image


def image_to_tensor(image, mean, std):
  image = input_transform(image, mean, std)
  image = image.transpose((2, 0, 1))
  image = torch.Tensor(image)
  return image.unsqueeze(0).cuda()


def gen_distance(shape):
  k = shape
  arr = torch.zeros([k * k, k * k], requires_grad=False)
  for i in range(k):
    for j in range(k):
      for x in range(k):
        for y in range(k):
          arr[i * k + j][x * k + y] = math.sqrt((i - x) * (i - x) + (j - y) * (j - y)) + 1

  return arr


def gen_cons_conv_weight(shape):
  center = int(shape / 2)
  accumulation = 0
  for i in range(shape):
    for j in range(shape):
      if i != center or j != center:
        dis = math.sqrt((i - center) * (i - center) + (j - center) * (j - center))
        accumulation += 1 / dis

  base = 1 / accumulation
  arr = torch.zeros((shape, shape), requires_grad=False)
  for i in range(shape):
    for j in range(shape):
      if i != center or j != center:
        dis = math.sqrt((i - center) * (i - center) + (j - center) * (j - center))
        arr[i][j] = base / dis
  arr[center][center] = -1

  return arr.unsqueeze(0).unsqueeze(0).repeat(3, 3, 1, 1)


