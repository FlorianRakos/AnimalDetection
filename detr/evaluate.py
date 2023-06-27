from datasets import get_coco_api_from_dataset
from datasets.coco_eval import CocoEvaluator
from tqdm.notebook import tqdm
import torch

import torchvision
import os
from transformers import DetrFeatureExtractor
import pytorch_lightning as pl
from transformers import DetrConfig, DetrForObjectDetection
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# can be edited to improve performance with pay-off of lower precision
torch.set_float32_matmul_precision("high")

img_folder = "../Bambi/data"
ann_folder = "../Bambi/data/annotations"

modelName = "facebook/detr-resnet-50"

lr=1e-4
lr_backbone=1e-5
weight_decay=1e-4
batch_size=16

# parameters 
import argparse
parser = argparse.ArgumentParser(description='Evaluate DETR model')
parser.add_argument('-a', '--all', action='store_true', help='evaluate all models in load directory')
parser.add_argument('-i', '--images', action='store_true', help='make result images')
args = parser.parse_args()

evaluateAll = args.all
makeImages = args.images

numResults = 20 # number of images to annotate with model predictions

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(ann_folder, "train-combined.json" if train else "test.json")
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


class Detr(pl.LightningModule):

     def __init__(self, lr, lr_backbone, weight_decay):
         super().__init__()

         self.model = DetrForObjectDetection.from_pretrained(modelName, 
                                                             num_labels=len(test_dataset.coco.getCatIds()),
                                                             ignore_mismatched_sizes=True)
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay

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



feature_extractor = DetrFeatureExtractor.from_pretrained(modelName)
#val_dataset = CocoDetection(img_folder=f'{img_folder}/val', feature_extractor=feature_extractor, train=False)
#val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size)
test_dataset = CocoDetection(img_folder=f'{img_folder}/test', feature_extractor=feature_extractor, train=False)
test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size)

base_ds = get_coco_api_from_dataset(test_dataset) # this is actually just calling the coco attribute

cats = test_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}


iou_types = ['bbox']
coco_evaluator = CocoEvaluator(base_ds, iou_types) # initialize evaluator with ground truths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#----------------- STATISTICS -----------------#


loadDir = "../Bambi/checkpoints/load"
files = os.listdir(loadDir)
files = [file for file in files if file.endswith(".ckpt")] # filter for files with .ckpt ending


if (evaluateAll):
  for file in files:
    print("\nEvaluating: ", file)
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    model = Detr.load_from_checkpoint(loadDir + "/" + file, lr=lr, lr_backbone=lr_backbone, weight_decay=weight_decay)
    
    model.cuda() # Ensure model weights are of type cuda float

    for idx, batch in enumerate(tqdm(test_dataloader)):
      # get the inputs
      pixel_values = batch["pixel_values"].to(device)
      pixel_mask = batch["pixel_mask"].to(device)
      labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

      # forward pass
      outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

      orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
      results = feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
      res = {target['image_id'].item(): output for target, output in zip(labels, results)}
      coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    print("Model " + file + " evaluated")


  exit()
else:

  model = Detr.load_from_checkpoint(loadDir + "/" + files[0], lr=lr, lr_backbone=lr_backbone, weight_decay=weight_decay)
  print("///////// Loaded model from: ", files[0], " /////////")
  model.cuda()   # Ensure model weights are of type cuda float
 

  if (not makeImages):
    for idx, batch in enumerate(tqdm(test_dataloader)):
      # get the inputs
      pixel_values = batch["pixel_values"].to(device)
      pixel_mask = batch["pixel_mask"].to(device)
      labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]   # these are in DETR format, resized + normalized

      # forward pass
      outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

      orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
      results = feature_extractor.post_process(outputs, orig_target_sizes)   # convert outputs of model to COCO api
      res = {target['image_id'].item(): output for target, output in zip(labels, results)}
      coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()



#----------------- VISUALIZATION -----------------#

if (not makeImages or evaluateAll):
  print("Only produce results if makeImages is True and evaluateAll is False")
  exit()

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes, image_id):

    colors = COLORS * 100
    yOffset = -6

    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=1))
        cl = p.argmax()
        text = f'{id2label[cl.item()]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin + yOffset, text, fontsize=10,
                bbox=dict(facecolor='yellow', alpha=0.1))
    plt.axis('off')
    plt.savefig("../results/Image-" + str(image_id) + "-P.png")


    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)

    annotations = test_dataset.coco.imgToAnns[image_id]

    for ann, c in zip(annotations, colors):
      label_box = ann['bbox']
      class_id = ann['category_id']
      label = id2label[class_id]
      ax = plt.gca()
      ax.add_patch(plt.Rectangle((label_box[0], label_box[1]), label_box[2], label_box[3],
                                  fill=False, color=c, linewidth=1))
      text = f'{label}'
      ax.text(label_box[0], label_box[1] + yOffset, text, fontsize=10,
                bbox=dict(facecolor='yellow', alpha=0.1))

    plt.axis('off')
    plt.savefig("../results/Image-" + str(image_id) + "-GT.png")




  

def visualize_predictions(image, outputs, threshold=0.5, keep_highest_scoring_bbox=False, image_id=None):
  # keep only predictions with confidence >= threshold
  probas = outputs.logits.softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > threshold
  if keep_highest_scoring_bbox:
    keep = probas.max(-1).values.argmax()
    keep = torch.tensor([keep])
  
  # convert predicted boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)
    
  # plot results
  plot_results(image, probas[keep], bboxes_scaled, image_id)



import time
start = time.time()

for  i in range(len(test_dataset)):
  pixel_values, target = test_dataset[i]

  pixel_values = pixel_values.unsqueeze(0).to(device)

  # forward pass to get class logits and bounding boxes
  outputs = model(pixel_values=pixel_values, pixel_mask=None)

  image_id = target['image_id'].item()
  image = test_dataset.coco.loadImgs(image_id)[0]
  image = Image.open(os.path.join(f'{img_folder}/test', image['file_name']))

  visualize_predictions(image, outputs, threshold=0.9, keep_highest_scoring_bbox=False, image_id=image_id)


end = time.time()
print("Time elapsed: " + str(end - start))
