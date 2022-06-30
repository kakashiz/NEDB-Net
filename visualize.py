import matplotlib.pyplot as plt

def display(image, mask_gt, mask_pred, figsize=(16,16)):
  if mask_gt is None:
    _, axes = plt.subplots(2, 1, figsize=figsize)
  else:
    _, axes = plt.subplots(3, 1, figsize=figsize)

  i = 0
  height, width = image.shape[:2]
  axes[i].set_ylim(height, 0)
  axes[i].set_xlim(0, width)
  axes[i].axis('off')
  axes[i].imshow(image)

  if mask_gt != None:
    i += 1
    height, width = mask_gt.shape[:2]
    axes[i].set_ylim(height, 0)
    axes[i].set_xlim(0, width)
    axes[i].axis('off')
    axes[i].imshow(mask_gt)

  i += 1
  height, width = mask_pred.shape[:2]
  axes[i].set_ylim(height, 0)
  axes[i].set_xlim(0, width)
  axes[i].axis('off')
  axes[i].imshow(mask_pred)

  plt.show()
