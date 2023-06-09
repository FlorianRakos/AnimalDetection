import torchvision
import os
from transformers import DetrFeatureExtractor
import numpy as np
import os
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import DetrConfig, DetrForObjectDetection
import torch
from pytorch_lightning import Trainer
from tqdm.notebook import tqdm
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse


torch.set_float32_matmul_precision("high")   # can be edited to improve performance with pay-off of lower precision

lr=1e-4
lr_backbone=1e-5
weight_decay=1e-4
batch_size=32
num_steps=15000   # 50000steps ~ 600 epochs
img_folder = "./Bambi/data"
ann_folder = "./Bambi/data/annotations"
checkpoint_dir = "./Bambi/checkpoints/resnet-50"
continue_training_dir = "./Bambi/checkpoints/continueTraining"
modelName = "facebook/detr-resnet-50"


parser = argparse.ArgumentParser(description='Train DETR model')
parser.add_argument('-l', '--large', action='store_true', help='train large resnet-101-model')
parser.add_argument('-c', '--continue-training', action='store_true', help='continue training from checkpoint')
args = parser.parse_args()
trainLarge = args.large
continueTraining = args.continue_training


if (trainLarge):
  print("///////// Training large resnet-101 model /////////")
  modelName = "facebook/detr-resnet-101"
  checkpoint_dir = "./Bambi/checkpoints/resnet-101"

if (continueTraining):
  print("///////// Continuing training from checkpoint /////////")
  files = os.listdir(continue_training_dir)
  files = [file[:-5] for file in files]   # remove .ckpt from string
  checkpoint_dir = checkpoint_dir + "/continue-" + files[0] + "/"

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(ann_folder, "train.json" if train else "validation.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target



feature_extractor = DetrFeatureExtractor.from_pretrained(modelName)

train_dataset = CocoDetection(img_folder=f'{img_folder}/train', feature_extractor=feature_extractor)
val_dataset = CocoDetection(img_folder=f'{img_folder}/val', feature_extractor=feature_extractor, train=False)


# print("Number of training examples:", len(train_dataset))
# print("Number of validation examples:", len(val_dataset))


# based on https://github.com/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb
image_ids = train_dataset.coco.getImgIds()
# let's pick a random image
image_id = image_ids[np.random.randint(0, len(image_ids))]
#print('Image n°{}'.format(image_id))
image = train_dataset.coco.loadImgs(image_id)[0]
image = Image.open(os.path.join(f'{img_folder}/train', image['file_name']))

annotations = train_dataset.coco.imgToAnns[image_id]
draw = ImageDraw.Draw(image, "RGBA")

cats = train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}

for annotation in annotations:
  box = annotation['bbox']
  class_idx = annotation['category_id']
  x,y,w,h = tuple(box)
  draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
  draw.text((x, y), id2label[class_idx], fill='white')


# prepares the batch data
def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size)
batch = next(iter(train_dataloader))


pixel_values, target = train_dataset[0]
#print(train_dataset[0])



class Detr(pl.LightningModule):

     def __init__(self, lr, lr_backbone, weight_decay):
          super().__init__()

          self.model = DetrForObjectDetection.from_pretrained(modelName, 
                                                              num_labels=len(id2label),
                                                              ignore_mismatched_sizes=True)
          self.lr = lr
          self.lr_backbone = lr_backbone
          self.weight_decay = weight_decay

          # Save checkpoints
          self.checkpoint_dir = checkpoint_dir
          self.checkpoint_callback = ModelCheckpoint(
          dirpath=self.checkpoint_dir,
          filename="detr-model-{epoch:03d}-{validation_loss:.2f}",
          save_top_k=10,
          monitor="validation_loss",
          save_last=True,  # Save the last model checkpoint
          every_n_epochs = 20 
          )

     def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

       return outputs
     
     def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       pixel_mask = batch["pixel_mask"]
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

     def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())

        print("Validationsloss: ", loss)
        return loss

     def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)
        
        return optimizer

     def train_dataloader(self):
        return train_dataloader

     def val_dataloader(self):
        return val_dataloader




if (continueTraining): 
  files = os.listdir(continue_training_dir)
  if len(files) > 1:
    raise Exception("Only one checkpoint file allowed int continueTraining folder!")
  model = Detr.load_from_checkpoint(os.path.join(continue_training_dir, files[0]), lr=lr, lr_backbone=lr_backbone, weight_decay=weight_decay)
else:
  model = Detr(lr=lr, lr_backbone=lr_backbone, weight_decay=weight_decay)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])



# -------------- Train Model -------------- #


trainer = Trainer(max_steps=num_steps, gradient_clip_val=0.1, default_root_dir = "./Bambi/checkpoints/default", callbacks=[model.checkpoint_callback], accelerator="gpu")
trainer.fit(model)



today = datetime.now()
dateStr = (today.strftime("%m/%d/")
      + today.strftime("%H:%M"))

dateStr = dateStr.replace("/", "-")



trainer.save_checkpoint(checkpoint_dir  +"/detr_model_epoch:"
      + str(trainer.current_epoch)
      + "_"
      + dateStr
      + ".ckpt")









