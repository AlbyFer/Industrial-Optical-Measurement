from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from skimage import measure

path_image_Train = "/Users/StarShipIV/Google_Drive/Progetti/CELdek_Vision/Images/IMG_2212.JPG"

im = Image.open(path_image_Train)
im.show()

r,g,b = im.getpixel()