import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader

from src.datasets.FuDataset import FuDataset
from src.utils.path import get_path
from src.model.core.fasterrcnnn import fasterrcnn_resnet50_fpn
import numpy as np


model = fasterrcnn_resnet50_fpn(pretrained=True)
print(model.roi_heads.box_predictor)

num_classes = 2

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
print(model.roi_heads.box_predictor)
![](../9.jpg)



def collate_fn(batch):
  return tuple(zip(*batch))

train_dataset = FuDataset(
        img_dir=get_path("D:\datasets\Fu\JPEGImages/"),
        txt=get_path("D:\datasets\Fu\\train.txt"),
        xml_dir=get_path("D:\datasets\Fu\Annotaions/")
    )

val_dataset = FuDataset(
        img_dir=get_path("D:\datasets\Fu\JPEGImages/"),
        txt=get_path("D:\datasets\Fu\\val.txt"),
        xml_dir=get_path("D:\datasets\Fu\Annotaions/")
    )

train_data_loader = DataLoader(
  train_dataset,
  batch_size=1,
  shuffle=False,
  num_workers=4,
  collate_fn=collate_fn
)

valid_data_loader =DataLoader(
  val_dataset,
  batch_size=1,
  shuffle=False,
  num_workers=4,
  collate_fn=collate_fn
)

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# move model to the right device
model.to(device)

# create an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# create a learning rate scheduler
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
lr_scheduler = None

# train it for 10 epochs
num_epochs = 10


import time
from tqdm import tqdm
#from tqdm.notebook import tqdm as tqdm

itr = 1

total_train_loss = []
total_valid_loss = []

losses_value = 0

for epoch in range(num_epochs):

  start_time = time.time()

  # train ------------------------------

  model.train()
  train_loss = []

  pbar = tqdm(train_data_loader, desc='let\'s train')
  for images, targets, image_ids in pbar:
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    loss_dict = model(images, targets)

    losses = sum(loss for loss in loss_dict.values())
    losses_value = losses.item()
    train_loss.append(losses_value)

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    pbar.set_description(f"Epoch: {epoch+1}, Batch: {itr}, Loss: {losses_value}")
    itr += 1

  epoch_train_loss = np.mean(train_loss)
  total_train_loss.append(epoch_train_loss)

  # update the learning rate
  if lr_scheduler is not None:
    lr_scheduler.step()

  # valid ------------------------------

  with torch.no_grad():
    valid_loss = []

    for images, targets, image_ids in valid_data_loader:
      images = list(image.to(device) for image in images)
      targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

      loss_dict = model(images, targets)

      losses = sum(loss for loss in loss_dict.values())
      loss_value = losses.item()
      valid_loss.append(loss_value)

  epoch_valid_loss = np.mean(valid_loss)
  total_valid_loss.append(epoch_valid_loss)

  # print ------------------------------

  print(f"Epoch Completed: {epoch+1}/{num_epochs}, Time: {time.time()-start_time}, "
        f"Train Loss: {epoch_train_loss} val loss:{total_valid_loss}")

torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')