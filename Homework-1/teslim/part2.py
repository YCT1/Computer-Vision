"""
Yekta Can Tursun, 150170105
"""

import numpy as np
import os
import cv2
import moviepy.editor as mpy
from matplotlib import pyplot as plt

background = cv2.imread("Malibu.jpg")
background_height = background.shape[0]
background_width = background.shape[1]
ratio = 360/background_height
background = cv2.resize(background,(int(background_width*ratio),360))
print(background.shape)


### HISTROGRAM CREATION ###
# In this part, we will create histogram for each cat frame 
# and we will sum to the list of each color

# Creation of histogram bins for each channel
r_hist_list = np.zeros((1,256))
g_hist_list = np.zeros((1,256))
b_hist_list = np.zeros((1,256))

# Traversing thorugh all frames
for i in range(0,179):
    # Reading a image of cat
    image = cv2.imread("cat/cat_"+str(i)+".png")

    # Eliminating the green screen
    foreground = np.logical_or(image[:,:,1]<180,image[:,:,0]>150)

    # Eliminating non zero values
    nonzero_x, nonzero_y = np.nonzero(foreground)
    nonzero_cat_values = image[nonzero_x,nonzero_y,:]

    # Creating Red Channel's histogram by using NP's histogram
    # Note that we need the list of color channel
    # We can transpose the nonzero_cat_values and can access the color channels
    r_hist,r_bins = np.histogram(nonzero_cat_values.T[0], bins=range(0,257))
    r_hist_list += r_hist

    # Green Channel
    g_hist,g_bins = np.histogram(nonzero_cat_values.T[1], bins=range(0,257))
    g_hist_list += g_hist

    # Blue Channel
    b_hist,b_bins = np.histogram(nonzero_cat_values.T[2], bins=range(0,257))
    b_hist_list += b_hist

# We get all frames all channels histogram, we need to average them by dividing
# frame number that is 180, also note that our channel list are vector so we need to tranpose them
plt.plot(r_hist_list.T/180,color="r")
plt.plot(g_hist_list.T/180,color="g")
plt.plot(b_hist_list.T/180,color="b")

plt.title("histogram of cat avarage") 
plt.savefig("part2-historgram.png")
plt.show()



### Custom histogram matching function
def hist_match(source, target):
    # Before we continue lets save the shape of the source
    sourceShape = source.shape

    # At that point we can make 1d version of source and target by flating of them
    source = source.ravel()
    template = target.ravel()

    # Let's get the unique values for the souce and its indices (we will use them for reconstraction)
    # Also we will need number of the occurence of these inque indicies
    # For this we can use NP's unique function
    s_values, s_indicies, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)


    # At that point we need to create Cumulative Distribution Function (CDF)
    # We can use NP's cumsum funtion and since we will normalize them by using last element of the CDF so that it will end by 1
    # We can cast values to float
    s_cdf = np.cumsum(s_counts, dtype=float)
    s_cdf = s_cdf / s_cdf[len(s_cdf)-1]
    t_cdf = np.cumsum(t_counts, dtype=float)
    t_cdf = t_cdf / t_cdf[len(t_cdf)-1]


    # At that point, we will interpolate values from source CDF to target CDF 
    # By using NP's magnificent Interpolation function
    # Then we will use source indicies for reconstraction of the image to 1D
    # In the final part by using reshape we will reconstruct image from flatten to 2D
    final_image = np.interp(s_cdf, t_cdf, t_values)[s_indicies].reshape(sourceShape)
    return final_image

### VIDEO FOR HIST MATCHED RIGHT CAT ###
# In this we have taken from part1,
# Only difference is that we do histogram mathcing in between backgorund and right cat

images_list = []

# For every pixel squecence we will repreat the whole proccess
for i in range(0,179):
    # Firstly, let's read the image
    image = cv2.imread("cat/cat_"+str(i)+".png")
    
    # Eliminating the green screen
    foreground = np.logical_or(image[:,:,1]<180,image[:,:,0]>150)

    # Eliminating non zero values
    nonzero_x, nonzero_y = np.nonzero(foreground)
    nonzero_cat_values = image[nonzero_x,nonzero_y,:]

    # Creating new frame by copying the background image
    new_frame = background.copy()

    # Adding first cat to left
    new_frame[nonzero_x,nonzero_y,:] = nonzero_cat_values

    # Adding second cat to right by using negative values of y cordinates
    # It will create a symetric version of the cat according to Y-axsis

    #### HISTOGRAM MATCHING #####
    new_frame[nonzero_x,-nonzero_y,:] = hist_match(nonzero_cat_values,background)

    # Switching the color channel for the movie py libary
    new_frame = new_frame[:,:,[2,1,0]]

    # Adding to frame list or image list
    images_list.append(new_frame)


# Video creation part
clip = mpy.ImageSequenceClip(images_list,fps=25)
audio = mpy.AudioFileClip("selfcontrol_part.wav").set_duration(clip.duration)

clip = clip.set_audio(audioclip=audio)
clip.write_videofile("part2_video.mp4", codec="libx264")

