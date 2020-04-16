from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from skimage import measure


path_image = "/Users/StarShipIV/Google_Drive/Progetti/CELdek_Vision/Images/IMG_22.JPG" # One used for prototyping

im = Image.open(path_image)
im.show()

im_grey= im.convert('L')
im_grey.show()

im_bin = 1*(np.array(im_grey) < 110)
plt.imshow(im_bin, interpolation='nearest')
plt.show()


imdilated = morphology.dilation(im_bin, np.ones((40,40)))
plt.imshow(imdilated, interpolation='nearest')
plt.show()

label_list = measure.label(imdilated)
feats = measure.regionprops(label_list)

# Pay attention to take the largest object in picture
maj_axis = feats[0].major_axis_length
min_axis = feats[0].minor_axis_length