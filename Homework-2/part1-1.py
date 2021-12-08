# Yekta Can Tursun #
# 150170105 #

import moviepy.video.io.VideoFileClip as mpy
import moviepy.editor as mpyeditor
import cv2

vid = mpy.VideoFileClip("shapes_video.mp4")

frame_count = vid.reader.nframes
video_fps = vid.fps

# Let's create a list where we can store the frames
frame_list = []

for i in range(frame_count):
    frame = vid.get_frame(i*1.0/video_fps)

    # Since it is salt and pepper noise
    # We should apply median filter
    filtered = median_blur= cv2.medianBlur(frame, 3)

    # Add filtered frame to the list
    frame_list.append(filtered)

# Create the video
clip = mpyeditor.ImageSequenceClip(frame_list,fps=video_fps)
clip.write_videofile("Videos/part1-1_video.mp4", codec="libx264")


