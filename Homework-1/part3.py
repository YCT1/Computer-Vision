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

# This functions preapres for the wrap matrix calculations
def changeOrder(array):
    pts1 = array.T[0:2].T
    pts1temp = pts1.copy()


    pts1[0] = pts1temp[1]
    pts1[3] = pts1temp[2]
    pts1[1] = pts1temp[0]
    pts1[2] = pts1temp[3]
    return pts1


# This is the utility function
# Where it ads two image with black background
# If there is a black inside the image, it is not looked good
def maskedAdding(background, source,x=0,y=0):
    rows,cols,channels = source.shape
    roi = background[x:rows+x, y:cols+y ]

    img2gray = cv2.cvtColor(source,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 5, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)


    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    img2_fg = cv2.bitwise_and(source,source,mask = mask)

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
    
    # Matrix preparatition for finding coefficents
    b= np.array([u.flatten()])
    # Final part adding 1 to end and reshaping to 3*3 matrix
    return np.append(np.linalg.pinv(A) @ b.T,1).reshape(3,3)


# Sorting for the planes
def sortPlane(plane):
        return plane[1]

# It gives planes to render order. According to list, it will be rendered orderly
def planeOrder(planesMatrix,planes):
    
    for i in range(0,len(planes)):
        planes[i] = [planes[i],planesMatrix[i]]

    planes.sort(key=sortPlane,reverse=True)
    for i in range(0, len(planes)):
        planes[i] = planes[i][0]

    return planes
    
    
# List of the frames
frame_list = []

# Reading all frames and writing to big 9*432*4*3 matrix (tensor?)
planes = readAllPlanes(9)

# Setting video lenght
video_length = planes.shape[1]

# Album list
albumlist =[
    "albums/album.png",
    "albums/album2.png",
    "albums/album3.png",
    "albums/album4.png",
    "albums/album.png",
    "albums/album2.png",
    "albums/album3.png",
    "albums/album4.png",
    "albums/album.png"
]

# Size of the albums should be defined in clock wise order
size_of_albums = np.float32([[0, 0], [256, 0], [0, 256], [256, 256]])

# Importing the cat with headphones
cat = cv2.imread("cat-headphones.png")[:,:,[2,1,0]]

# Resize to desired background ratio
cat = cv2.resize(cat,[326,322])
for i in range(0,video_length):

    # Create background
    background = np.zeros([322,572,3])
    # Fill with 255, (make it to white)
    background.fill(255)
    
    
    # frame render list, in the end, items will be rendered according to it
    frame = []

    # Read all albums
    for name_of_album_i in range(0,len(albumlist)):
        
        # Reading the album
        album = cv2.imread(albumlist[name_of_album_i])

        # Finding the perspective matrix and note that we are using changeOrder for each point so that it will be aligned to matrix calculation
        matrix = findPerspectiveMatrix(changeOrder(planes[name_of_album_i][i]).astype(np.float32), size_of_albums.astype(np.float32))

        # Since it is inverse operation, we need to take inverse of it
        matrix = np.linalg.inv(matrix)

        # warp the perspective
        result = cv2.warpPerspective(album, matrix, (572, 322))[:,:,[2,1,0]]

        # Add to render list
        frame.append([result,False])
    
    # Calculating the render list order according to items
    # Matrix for the depth calculationm
    depthPlaneMatrix = np.zeros([10])
    for k in range(0,planes.shape[0]):
        # Take the awerage depth of the 
        depthPlaneMatrix[k] = np.amax(planes[k][i].T[2:3].T)

    depthPlaneMatrix[9] = 278  
    frame.append([cat,True]) 
    frame = planeOrder(depthPlaneMatrix, frame)
    
    for i in range(0,len(frame)):
        #print(frame[i][0].shape)
        if frame[i][1]:
            background = maskedAdding(background,frame[i][0],y=125)
        else:
            background = maskedAdding(background,frame[i][0], y=0)

    frame_list.append(background)
    #cv2.imshow("test", (background/256)[:,:,[2,1,0]])
    #cv2.waitKey(0)
    #exit()

clip = mpy.ImageSequenceClip(frame_list,fps=25)
clip.write_videofile("part3_video.mp4", codec="libx264")