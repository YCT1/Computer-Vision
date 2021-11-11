import numpy as np
import os
import cv2
import moviepy.editor as mpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# This function reads the planes file and returns result as matrix
def readFile(address):
    f = open(address, "r")
    s = f.read()
    s = s.split("\n")
    s = s[:-1]
    s_array = []
    for x in s:
        splited = x.split(")(")
        for i in range(0,len(splited)):
            splited[i] = splited[i].replace("(","")
            splited[i] = splited[i].replace(")","")
            splited[i] = np.array(splited[i].split(" ")).astype(np.float64)
        s_array.append(np.array(splited).astype(np.float64))

    return np.array(s_array).astype(np.float64)

# This function reads the planes by their adress
def readAllPlanes(number_of_planes, plane_name = "Plane_", file_format=".txt"):
    planes = []
    for i in range(1,number_of_planes+1):
        planes.append(readFile(plane_name + str(i) + file_format))
    return np.array(planes).astype(np.float64)


def changeOrder(array):
    pts1 = array.T[0:2].T
    pts1temp = pts1.copy()


    pts1[0] = pts1temp[1]
    pts1[3] = pts1temp[2]
    pts1[1] = pts1temp[0]
    pts1[2] = pts1temp[3]
    return pts1

planes = readAllPlanes(9)

video_length = planes.shape[1]

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
"""
frame_list = []
album = cv2.imread("albums/album.png")
album2 = cv2.imread("albums/album2.png")
pts2 = np.float32([[0, 0], [256, 0], [0, 256], [256, 256]])
for i in range(0,video_length):
    matrix1 = cv2.getPerspectiveTransform(changeOrder(planes[0][i]).astype(np.float32), pts2.astype(np.float32))
    matrix1 = np.linalg.inv(matrix1)

    matrix2 = cv2.getPerspectiveTransform(changeOrder(planes[1][i]).astype(np.float32), pts2.astype(np.float32))
    matrix2 = np.linalg.inv(matrix2)

    result = cv2.warpPerspective(album, matrix1, (572, 322))[:,:,[2,1,0]]
    resutl2 = cv2.warpPerspective(album2, matrix2,(572, 322))[:,:,[2,1,0]]
    r= cv2.add(result,resutl2)
    frame_list.append(r)

"""

frame_list = []
# Generilization
albumlist =[
    "albums/album.png",
    "albums/album2.png",
    "albums/album3.png",
    "albums/album.png",
    "albums/album2.png",
    "albums/album3.png",
    "albums/album.png",
    "albums/album2.png",
    "albums/album3.png"
]
"""
size_of_albums = np.float32([[0, 0], [256, 0], [0, 256], [256, 256]])
for i in range(0,video_length):
    background = np.zeros([322,572,3])
    background.fill(255)
    frame = []
    for name_of_album_i in range(0,len(albumlist)):
        album = cv2.imread(albumlist[name_of_album_i])
        matrix = cv2.getPerspectiveTransform(changeOrder(planes[name_of_album_i][i]).astype(np.float32), size_of_albums.astype(np.float32))
        matrix = np.linalg.inv(matrix)
        result = cv2.warpPerspective(album, matrix, (572, 322))[:,:,[2,1,0]]
        frame.append(result)
    
    background = cv2.add(background.astype(np.float32),frame[0].astype(np.float32))
    
    for i in range(1,len(frame)):
        background = cv2.add(frame[i].astype(np.float32),background.astype(np.float32))
    frame_list.append(background)


clip = mpy.ImageSequenceClip(frame_list,fps=25)
clip.write_videofile("part3_video.mp4", codec="libx264")
"""

size_of_albums = np.float32([[0, 0], [256, 0], [0, 256], [256, 256]])
for i in range(0,video_length):
    background = np.zeros([322,572,3])
    background.fill(255)
    frame = []
    for name_of_album_i in range(0,len(albumlist)):
        album = cv2.imread(albumlist[name_of_album_i])
        matrix = cv2.getPerspectiveTransform(changeOrder(planes[name_of_album_i][i]).astype(np.float32), size_of_albums.astype(np.float32))
        matrix = np.linalg.inv(matrix)
        result = cv2.warpPerspective(album, matrix, (572, 322))[:,:,[2,1,0]]
        frame.append(result)
    
    
    
    for i in range(0,len(frame)):
        background = maskedAdding(background,frame[i])

    frame_list.append(background)


clip = mpy.ImageSequenceClip(frame_list,fps=25)
clip.write_videofile("part3_video.mp4", codec="libx264")