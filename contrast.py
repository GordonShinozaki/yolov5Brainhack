#microservice for improving training data
import cv2
import numpy as np

#use this to increase contrast in photos for ease of feature extraction
def contrast_up(filename):
  img = cv2.imread(foldername + filename, 1) #Change filename here - where is your data?
  # converting to LAB color space
  lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  l_channel, a, b = cv2.split(lab)

  # Applying CLAHE to L-channel
  # feel free to try different values for the limit and grid size:
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  cl = clahe.apply(l_channel)

  ##we can also try the below for an affine transform, but no need
  ##new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)

  # merge the CLAHE enhanced L-channel with the a and b channel
  limg = cv2.merge((cl,a,b))

  # Converting image from LAB Color model to BGR color spcae
  enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

  #then write into a new file, with same name
  cv2.imwrite(foldername+filename, enhanced_img)

