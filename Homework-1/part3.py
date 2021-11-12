import numpy as np
import os
import cv2
import moviepy.editor as mpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.ma import masked


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



def maskedAdding(background, source,x=0,y=0):
    rows,cols,channels = source.shape
    roi = background[0:rows, 0:cols ]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(source,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 5, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(source,source,mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg.astype(np.float32),img2_fg.astype(np.float32))
    background[x:rows+x, y:cols+y ] = dst
    return background


# Perspective Matrix Calculator
def findPerspectiveMatrix(x = np.zeros([4,2]), u = np.zeros([4,2])):
    c = np.zeros([3,3])
    c[2][2] = 1
    x = x.T
    u = u.T

    # Holy Matrix

    A = np.array([
        [x[0][0], x[1][0], 1, 0, 0, 0, -x[0][0]*u[0][0], -x[1][0]*u[0][0]],
        [x[0][1], x[1][1], 1, 0, 0, 0, -x[0][1]*u[0][1], -x[1][1]*u[0][1]],
        [x[0][2], x[1][2], 1, 0, 0, 0, -x[0][2]*u[0][2], -x[1][2]*u[0][2]],
        [x[0][3], x[1][3], 1, 0, 0, 0, -x[0][3]*u[0][3], -x[1][3]*u[0][3]],
        [0, 0, 0, x[0][0], x[1][0], 1, -x[0][0]*u[1][0], -x[1][0]*u[1][0]],
        [0, 0, 0, x[0][1], x[1][1], 1, -x[0][1]*u[1][1], -x[1][1]*u[1][1]],
        [0, 0, 0, x[0][2], x[1][2], 1, -x[0][2]*u[1][2], -x[1][2]*u[1][2]],
        [0, 0, 0, x[0][3], x[1][3], 1, -x[0][3]*u[1][3], -x[1][3]*u[1][3]],
    ])
    
    b= np.array([u.flatten()])
    return np.append(np.linalg.pinv(A) @ b.T,1).reshape(3,3)


def sortPlane(plane):
        return plane[1]

def planeOrder(planesMatrix,planes):
    
    for i in range(0,len(planes)):
        planes[i] = [planes[i],planesMatrix[i]]

    planes.sort(key=sortPlane,reverse=True)
    for i in range(0, len(planes)):
        planes[i] = planes[i][0]

    return planes
    
    

frame_list = []
planes = readAllPlanes(9)
video_length = planes.shape[1]
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


size_of_albums = np.float32([[0, 0], [256, 0], [0, 256], [256, 256]])
cat = cv2.imread("cat-headphones.png")[:,:,[2,1,0]]
cat = cv2.resize(cat,[326,322])
for i in range(0,video_length):
    background = np.zeros([322,572,3])
    background.fill(255)
    background = maskedAdding(background,cat,y=125)
    frame = []
    for name_of_album_i in range(0,len(albumlist)):
        album = cv2.imread(albumlist[name_of_album_i])
        matrix = findPerspectiveMatrix(changeOrder(planes[name_of_album_i][i]).astype(np.float32), size_of_albums.astype(np.float32))
        matrix = np.linalg.inv(matrix)
        result = cv2.warpPerspective(album, matrix, (572, 322))[:,:,[2,1,0]]
        frame.append(result)
    
    depthPlaneMatrix = np.zeros([9])
    for k in range(0,planes.shape[0]):
        depthPlaneMatrix[k] = np.mean(planes[k][i].T[:3].T)

       
    frame = planeOrder(depthPlaneMatrix, frame)
    for i in range(0,len(frame)):
        background = maskedAdding(background,frame[i])

    frame_list.append(background)


clip = mpy.ImageSequenceClip(frame_list,fps=25)
clip.write_videofile("part3_video.mp4", codec="libx264")