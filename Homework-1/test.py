import numpy as np
import os
import cv2
import moviepy.editor as mpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.ma import masked
def maskedAdding(background, source):
    rows,cols,channels = source.shape
    roi = background[0:rows, 0:cols ]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(source,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(source,source,mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg.astype(np.float32),img2_fg.astype(np.float32))
    background[0:rows, 0:cols ] = dst
    return background
new_background = np.zeros([256,256,3])
album = cv2.imread("albums/album.png")

new_background.fill(255)
new_background = maskedAdding(new_background,album)

cv2.imshow("tes",new_background)
cv2.waitKey(0)