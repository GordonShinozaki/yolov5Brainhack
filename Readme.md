# Yolov5 CV model
These code bunches are what we've used for 2022 CDDC brainhack, finishing 5th in the finals. 
This is based on the Yolov5 model produced on github.

Since the code was originally written on colab, the ipynb is included for ease of access. However, we've removed all our training data. 

## Potential Training improvements
A higher contrasted photo increases ease of feature extraction, perhaps consider our contrast.py script to improve photo contrast

## Training the model
python train.py --img 1280 --batch -1 --epochs 30 --data dataset.yaml --weights yolov5m.pt
## Tuning the model
python train.py --img 640 --batch 16 --epochs 5 --data ./data/dataset.yaml --weights ./runs/train/exp40/weights/last.pt --evolve 10
## Inference
Refer to our test_inference script

