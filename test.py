import cv2
import torch
import visualize
import NEDBNet as net
import function
from config import Config

if __name__ == '__main__':
  Config.align_corners = False
  model = net.get_seg_model()
  model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
  function.load_pre_weights(model, "weights.pth")

  image_path = ""
  image = cv2.imread(image_path)
  mask_pred = function.inference(model, image)
  visualize.display(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), None, mask_pred)


