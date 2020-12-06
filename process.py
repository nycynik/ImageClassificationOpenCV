import numpy as np
import cv2

# read in the models
model_data = open('./models/synset_words.txt').read().strip().split('\n')

classes = [r[r.find(' ') + 1:] for r in model_data]

net = cv2.dnn.readNetFromCaffe(
    './models/bvlc_googlenet.prototxt',
    './models/bvlc_googlenet.caffemodel')

# read the image
images = []
images.append(cv2.imread('./images/woman-beach-sea-ocean-clam-search.jpg'))
images.append(cv2.imread('./images/woman-hiking.jpg'))

# make the blob
#blob = cv2.dnn.blobFromImages(images, 1, (224, 224))
blob = cv2.dnn.blobFromImage(images[1], 1, (224, 224))

# here we go!
net.setInput(blob)

# forward only - get inference
output = net.forward()

# show top 5
index = np.argsort(output[0])[::-1][:5]

print(f'Rank\tID\tLikely\tDescription')
for i, id in enumerate(index):
    print(f'{i+1}\t{id}\t{(output[0][id] * 100):.3}%\t{classes[id]}')
