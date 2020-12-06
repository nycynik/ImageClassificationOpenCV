import numpy as np
import cv2

# read in the models
model_data = open('./models/synset_words.txt').read().strip().split('\n')

classes = [r[r.find(' ') + 1:] for r in model_data]

net = cv2.dnn.readNetFromCaffe(
    './models/bvlc_googlenet.prototxt',
    './models/bvlc_googlenet.caffemodel')

# read the image
img = cv2.imread('./images/woman-beach-sea-ocean-clam-search.jpg')
print(img.shape)

# make the blob
blob = cv2.dnn.blobFromImage(img, 1, (224, 224))

# here we go!
net.setInput(blob)
