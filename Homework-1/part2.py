import numpy as np
import os
import cv2
import moviepy.editor as mpy
from matplotlib import pyplot as plt

background = cv2.imread("Malibu.jpg")
#cv2.imshow("Background Image Window", background)
#cv2.waitKey(0)

background_height = background.shape[0]
background_width = background.shape[1]

ratio = 360/background_height

background = cv2.resize(background,(int(background_width*ratio),360))

print(background.shape)


images_list = []

image = cv2.imread("cat/cat_"+str("5")+".png")
foreground = np.logical_or(image[:,:,1]<180,image[:,:,0]>150)
nonzero_x, nonzero_y = np.nonzero(foreground)
nonzero_cat_values = image[nonzero_x,nonzero_y,:]
new_frame = background.copy()
new_frame[nonzero_x,nonzero_y,:] = nonzero_cat_values
new_frame[nonzero_x,-nonzero_y,:] = nonzero_cat_values
#new_frame = new_frame[:,:,[2,1,0]]

r_hist_list = np.zeros(256)
g_hist_list = np.zeros(256)
b_hist_list = np.zeros(256)
for i in range(0,179):
    image = cv2.imread("cat/cat_"+str(i)+".png")
    foreground = np.logical_or(image[:,:,1]<180,image[:,:,0]>150)
    nonzero_x, nonzero_y = np.nonzero(foreground)
    nonzero_cat_values = image[nonzero_x,nonzero_y,:]
    r_hist,r_bins = np.histogram(nonzero_cat_values.T[0], bins=range(0,257))
    r_hist_list += r_hist

    g_hist,g_bins = np.histogram(nonzero_cat_values.T[1], bins=range(0,257))
    g_hist_list += g_hist

    b_hist,b_bins = np.histogram(nonzero_cat_values.T[2], bins=range(0,257))
    b_hist_list += b_hist


plt.plot(r_hist_list/180,color="r")
plt.plot(g_hist_list/180,color="g")
plt.plot(b_hist_list/180,color="b")

plt.title("histogram of cat avarage") 
plt.show()



### Custom histogram matching function
def hist_match(source, target):
    oldshape = source.shape
    source = source.ravel()
    template = target.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)