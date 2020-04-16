from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from skimage import measure

path_image_Train = "/Users/StarShipIV/Google_Drive/Progetti/CELdek_Vision/Images/IMG_2209.JPG"

im = Image.open(path_image_Train)
im.show()

im_grey= im.convert('L')
im_grey.show()

im_bin = 1*(np.array(im_grey) < 155)
plt.imshow(im_bin, interpolation='nearest')
plt.show()

im_eroded = morphology.erosion(im_bin, np.ones((5,5)))
im_dilated = morphology.dilation(im_eroded, np.ones((5,5)))
plt.imshow(im_dilated, interpolation='nearest')
plt.show()

label_list = measure.label(im_dilated)
feats = measure.regionprops(label_list)

maj_axis_pix = feats[0].major_axis_length
min_axis_pix = feats[0].minor_axis_length

# pix_to_mm = maj_axis_pix/199     #it's 9.487575323750026 pixels/mm

maj_axis_mm = round(maj_axis_pix/9.487575323750026, 3)
min_axis_mm = round(min_axis_pix/9.487575323750026, 3)

print("Dimensions", maj_axis_mm, " x ", min_axis_mm)

# ------------------------------------------------------------------------------------------------------- #
# Testing

from PIL import Image
import numpy as np
from skimage import morphology
from skimage import measure

path_image_Test= "/Users/StarShipIV/Google_Drive/Progetti/CELdek_Vision/Images/IMG_2210.JPG"

im = Image.open(path_image_Train)
im_grey= im.convert('L')
im_bin = 1*(np.array(im_grey) < 155)
plt.imshow(im_bin, interpolation='nearest')

im_eroded = morphology.erosion(im_bin, np.ones((5,5)))
im_dilated = morphology.dilation(im_eroded, np.ones((5,5)))
label_list = measure.label(im_dilated)
feats = measure.regionprops(label_list)

maj_axis_pix = feats[0].major_axis_length
min_axis_pix = feats[0].minor_axis_length

maj_axis_mm = round(maj_axis_pix/9.487575323750026, 3)
min_axis_mm = round(min_axis_pix/9.487575323750026, 3)

print("Dimensions", maj_axis_mm, " x ", min_axis_mm)