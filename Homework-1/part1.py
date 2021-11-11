"""
Yekta Can Tursun, 150170105
"""


import numpy as np
import os
import cv2
import moviepy.editor as mpy


background = cv2.imread("Malibu.jpg")

background_height = background.shape[0]
background_width = background.shape[1]

ratio = 360/background_height

background = cv2.resize(background,(int(background_width*ratio),360))

print(background.shape)

# A list of image will be used as frames for the video
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
    new_frame[nonzero_x,-nonzero_y,:] = nonzero_cat_values

    # Switching the color channel for the movie py libary
    new_frame = new_frame[:,:,[2,1,0]]

    # Adding to frame list or image list
    images_list.append(new_frame)


# Video creation part
clip = mpy.ImageSequenceClip(images_list,fps=25)
audio = mpy.AudioFileClip("selfcontrol_part.wav").set_duration(clip.duration)

clip = clip.set_audio(audioclip=audio)
clip.write_videofile("part1_video.mp4", codec="libx264")

