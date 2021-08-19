import numpy as np
from PIL import Image

def load_image(self, path):
    img = Image.open(path)
    img = img.convert('RGB')
    return img



# self.images = [np.asarray(self.load_image(image_path))/255 for image_path in self.imagenames[:30000]]
self.images = [np.asarray(self.load_image(image_path))/255 for image_path in self.imagenames]
mean_val = [np.mean(img, axis=tuple(range(img.ndim-1))) for img in self.images] # calc for RGB at once
mean_dataset = np.mean(np.vstack(mean_val),axis=0)
print(mean_dataset)

std_val = [np.std(img, axis=tuple(range(img.ndim-1))) for img in self.images]
std_dataset = np.std(np.vstack(std_val), axis=0)
print(std_dataset)



"""
imagenet : transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
training set: [Matt]
                        [0.13804892 0.21836744 0.20237076]
                            [0.08498618 0.07658653 0.07137364]

102350 items

Eileen
[0.10853149 0.1889904  0.20464881]
[0.06907555 0.07234943 0.0738185 ]

cages:
                             [0.08940355 0.185837   0.19124651]
                             [0.05299317 0.05526755 0.05529101]

=========================================

[0.13804892 0.21836744 0.20237076]
[0.08498618 0.07658653 0.07137364]


[0.12977006 0.208491   0.19187527]
[0.08484888 0.07647699 0.07088686]

[33.09136404 53.16520591 48.92819371]
[21.63646478 19.50163343 18.07615005]
"""