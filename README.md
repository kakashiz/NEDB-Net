# NEDB-Net
Code for the paper "Noise and Edge Based Dual Branch Image Manipulation Detection"

# Environment
Python 3.6, Pytorch 1.8.   
Other packages can reference `requirements.txt`.

# Model weights
The model trained with CASIAv2 dataset images was uploaded in Google Drive: 
https://drive.google.com/file/d/1f3yMCOm2U5QMVdgfZjdRxzxPkfLxz3RP/view?usp=sharing

# Train
The training list of CASIAv2 is shown in `train.txt`. Note that if an image file is named 
`Tp_D_NRD_S_N_sec00001_cha00042_00001.tif`, we will rename it to `00001.tif`.  
It is worth noting that there are non-reproducible functions in pytorch such as `torch.nn.functional.interpolate`. 
In order to be able to obtain approximate the results, please call the following functions before training:
```
def set_seed(seed=2022):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
  np.random.seed(seed)  # Numpy module.
  random.seed(seed)  # Python random module
  os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
  os.environ['PYTHONHASHSEED'] = str(seed)
```
The random number seed is set to `2022`. In addition, the following settings are also required in the Dataloader:
```
def worker_init_fn(worker_id):
  worker_seed = torch.initial_seed() % 2 ** 32
  np.random.seed(worker_seed)
  random.seed(worker_seed)
  
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
```

# Test
Please set the test image path in `test.py`, then run `test.py` with Python.


