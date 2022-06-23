import torch
import utils
display = utils.notebook_init()  # checks

"""Create labels from annotations.
Labels format: Class x_middle y_middle width height
"""

import json
jsonfile = open('training_data_new.json')

data = json.load(jsonfile)
for i in data['images']:
  for j in data['annotations']:
    if j['image_id'] == i['id']:
      imagehash, jpg = i['file_name'].split(".")
      txtFileName = imagehash + ".txt"
      height = i['height']
      width = i['width']
      bbox = j['bbox']
      category = j['category_id']
      x_min = bbox[0]
      y_min = bbox[1]
      x_dir = bbox[2]
      y_dir = bbox[3]
      x_center = x_min + x_dir // 2
      y_center = y_min + y_dir // 2
      normalise_x_center = x_center / width
      normalise_y_center = y_center / height
      normalise_x_dir = x_dir / width
      normalise_y_dir = y_dir / height

      f = open(f"./labels/{txtFileName}", "a")
      if category == 1:
        f.write(f"0 {normalise_x_center} {normalise_y_center} {normalise_x_dir} {normalise_y_dir} \n")
      if category == 2:
        f.write(f"1 {normalise_x_center} {normalise_y_center} {normalise_x_dir} {normalise_y_dir} \n")
      f.close()

"""Train model
--img will choose image size
--batch -1 will auto batch to get biggest batch based on GPU memory
--epochs will choose number of epochs
--data choose .yaml file containing routes to images and labels. Also state number of classes
--weights choose pretrained weights


Final weights will be saved in runs/train/exp{x}/weights (x is the run number)

Pre-trained weights are auto downloaded.

arguments used for our model:
!python train.py --img 640 --batch 16 --epochs 150 --data dataset.yaml --weights yolov5x.pt
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd ./drive/MyDrive/yolov5

#run this code 
#python train.py --img 1280 --batch -1 --epochs 30 --data dataset.yaml --weights yolov5m.pt