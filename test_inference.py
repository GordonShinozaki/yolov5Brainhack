##This is the microservice to predict and generate model results

exp_number = "40" #change here for newer runs, we track this number based on which exp_number gives the best results.

import torch

#import the self-trained yolov5 model that is stored locally
#we built this model based off this open source model here: https://github.com/ultralytics/yolov5
model = torch.hub.load('yolov5', 'custom', path="yolov5/runs/train/exp"+str(exp_number)+"/weights/last.pt", source='local')

##Parse Json file and get all image names

import json

f = open("/content/drive/My Drive/CVFinal/qualifiers_finals_no_annotations.json")

root = "/content/drive/MyDrive/CVFinal/Images/"

data = json.load(f)

images = [] #imagine this as a set of lists. [0] = id, [1] = image path 

for image in data['images']:
    image_data =[image["id"], root+image["file_name"]]
    images.append(image_data)

len(images)

#initialize list of objects to be dumped as json later
json_objects = []

#this is a confidence threshold that we define, below which the item predicted 
#will not be accepted as a valid prediction
confidence_threshold = 0.2 #change here for threshold

for image in images:
  #get prediction results and parse it into a dataframe
  #turn augment on, which will take more time to infer, but increases acc
  results = model(image[1], augment = True)
  df = results.pandas().xyxy[0]

  #filter df by threshold
  rslt_df = df[df['confidence'] > confidence_threshold] 
  #run operation for each row in df

  #if the length of the result dataframe is 0, then the model 
  #predicted no objects. We need to hardcore a null object in this case
  if len(rslt_df) == 0:
    json_object = {
        "image_id": image[0],
        "bbox": [0.0000, 0.0000, 1.0, 1.0],
        "category_id": 1,
        "score": 0.99,
      }
    json_objects.append(json_object)
  else:
    for i in range(0, len(rslt_df)):
      #calculate box width by xmax - xmin
      width = round(rslt_df.iloc[i]["xmax"]-rslt_df.iloc[i]["xmin"],1) 
      #calculate box height ymax - ymin
      height = round(rslt_df.iloc[i]["ymax"]-rslt_df.iloc[i]["ymin"],1)
      #xmin as is, but round to 1dp
      x_min = round(rslt_df.iloc[i]["xmin"],1)
      #xmax as is, but round to 1dp
      y_min = round(rslt_df.iloc[i]["ymin"],1)
      confidence = round(rslt_df.iloc[i]["confidence"], 6)
      #because it's 0 and 1, just + 1 to it and we'll be fine
      category = int(rslt_df.iloc[i]["class"]) + 1
      json_object = {
          "image_id": image[0],
          "bbox": [x_min, y_min, width, height],
          "category_id": category,
          "score": confidence,
      }
      json_objects.append(json_object)

len(json_objects)

#dump json into a file
json_list = json.dumps(json_objects, indent=4)
print(json_list)
with open("score.json", "w") as outfile:
  outfile.write(json_list)

