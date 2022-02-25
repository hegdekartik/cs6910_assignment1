import os
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(img, lab), (timg, tlab) = fashion_mnist.load_data()
class_names = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(5,5,i+1)
    # print(lab[i])
    plt.axis("off")
    plt.imshow(img[i])
    plt.title(class_names[lab[i]])
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()