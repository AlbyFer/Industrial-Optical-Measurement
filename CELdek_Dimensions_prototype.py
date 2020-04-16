
# Pictures taken with Iphone5 at 28 cms of distance from object. Perpendicular to main object surface.

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from skimage import measure
import matplotlib.patches as patches


path_image = "/Users/StarShipIV/Google_Drive/Progetti/CELdek_Vision/Images/IMG_2041.JPG" # One used for prototyping
# path_image = "/Users/StarShipIV/Google_Drive/Progetti/CELdek_Vision/Images/IMG_2043.JPG" # Test



im = Image.open(path_image)
im.show()

im_grey= im.convert('L')
im_grey.show()

im_bin = 1*(np.array(im_grey) < 100)
plt.imshow(im_bin, interpolation='nearest')
plt.show()


imeroded = morphology.erosion(im_bin, np.ones((10,10)))
imdilated = morphology.dilation(im_bin, np.ones((10,10)))
label_list = measure.label(imdilated)
feats = measure.regionprops(label_list)

plt.imshow(label_list, interpolation='nearest')
plt.show()

maj_axis = feats[0].major_axis_length
min_axis = feats[0].minor_axis_length

length = maj_axis * 0.008389958386775592
height = min_axis * 0.008509512658515881

print(length, height)

rect = patches.Rectangle((feats[0].local_centroid),maj_axis,min_axis,linewidth=1,edgecolor='r',facecolor='none')
im.add_patch(rect)

plt.imshow(feats[0].filled_image, interpolation='nearest')
plt.show()
