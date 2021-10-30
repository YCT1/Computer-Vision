import numpy as np
import os
import cv2
import moviepy.editor as mpy


background = cv2.imread("Malibu.jpg")
#cv2.imshow("Background Image Window", background)
#cv2.waitKey(0)

background_height = background.shape[0]
background_width = background.shape[1]

ratio = 360/background_height

background = cv2.resize(background,(int(background_width*ratio),360))

print(background.shape)


images_list = []



for i in range(0,179):
    image = cv2.imread("cat/cat_"+str(i)+".png")
    foreground = np.logical_or(image[:,:,1]<180,image[:,:,0]>150)
    nonzero_x, nonzero_y = np.nonzero(foreground)
    nonzero_cat_values = image[nonzero_x,nonzero_y,:]
    new_frame = background.copy()
    new_frame[nonzero_x,nonzero_y,:] = nonzero_cat_values
    new_frame[nonzero_x,-nonzero_y,:] = nonzero_cat_values
    new_frame = new_frame[:,:,[2,1,0]]
    images_list.append(new_frame)



clip = mpy.ImageSequenceClip(images_list,fps=25)
audio = mpy.AudioFileClip("selfcontrol_part.wav").set_duration(clip.duration)

clip = clip.set_audio(audioclip=audio)
clip.write_videofile("part1_video.mp4", codec="libx264")

