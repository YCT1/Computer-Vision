import numpy as np
import os
import cv2
import moviepy.editor as mpy
import re
import glob

background = cv2.imread('Malibu.jpg')

# background has a shape pf (776, 1998, 3) and we need to
# resize the background image according to given frames.
# Since cat frames has shapes of (360,640,3);
# we will multiply 'ratio' value with both height and width

background_height = background.shape[0]
background_width = background.shape[1]
ratio = 360 / background_height

# in cv2's resize function, after the file paremeter, we need to
# give scaled size values.(Fx,Fy). Fx denotes the scale along the
# horizontal access and Fy for vertical access
# So, Fx -> width , Fy -> height
# cv2.resize(file,(Fx,Fy))

background = cv2.resize(background,(int(background_width*ratio),360))

main_dir = 'cat' # main directory. we can also get it by os.getcwd()
images_list = []

parent_dir = os.getcwd()
cat_image_dir = parent_dir + '\cat' 
os.chdir(cat_image_dir) # # changing working directory in order to easily access images
 
#!!!!for file in os.listdir(): # until all png files being read
# we are not using os.listdir() because it gets files from folder randomly
    
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
for file in sorted(glob.glob('*.png'),key=numericalSort):
    
    image = cv2.imread(file)
    
    # we find pixels having cat images with distict them with green background
    foreground = np.logical_or(image[:,:,1] < 180, image[:,:,0] > 150)
    # now, foreground is (360,640) shaped and its values are 
    # consisting of only True, False.
    
    # If we know locations of true values, then we can evaluate
    # pixels whose values belonging to the cat part.
    
    nonzero_x, nonzero_y = np.nonzero(foreground)
    # nonzero_x contains index values of rows of the matrix and y is for columns.
    # nonzero_x like [0 0 0 ... 359 359 359]
    # nonzero_x and nonzero_y has same logic
    # Actually, nonzero_x and nonzero_y has shape of (94061,) which
    # means between (360x640) pixels, there are 94061 pixels that are
    # belonging the cat part
    
    nonzero_cat_values = image[nonzero_x,nonzero_y,:] # has shape of (94061,3)
    
    new_frame = background.copy()
    new_frame[nonzero_x,nonzero_y,:] = nonzero_cat_values
    new_frame = new_frame[:,:,[2,1,0]]
    images_list.append(new_frame)

os.chdir(parent_dir) # now, it is in parent folder
clip = mpy.ImageSequenceClip(images_list, fps = 25)
audio = mpy.AudioFileClip('selfcontrol_part.wav').set_duration(clip.duration)
clip = clip.set_audio(audioclip=audio)
clip.write_videofile('part1_video_kerem.mp4', codec = 'libx264')

    