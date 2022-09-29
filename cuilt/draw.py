import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt

coordinate = "1030 1181 , 1110 1195 , 826 640 , 1569 747 , 1609 787 , 1609 793 , 1569 833 , 680 1273 , 995 1438 , 1142 1438 , 1569 1273 , 1609 1313 , 1609 1319 , 1569 1359 , 1091 849"

coordinate = coordinate.split(",")
# print(coordinate)

# print(coordinate[0].split(" "))
for i in range(len(coordinate)):
    coordinate[i] = coordinate[i].split(" ")
    coordinate[i] = list(filter(None, coordinate[i]))
    for j in range(len(coordinate[i])):
        coordinate[i][j] = int(coordinate[i][j])
# coordinate = [[971, 1008]]
# print(coordinate)
img = Image.open("target_binary/M1_test9.png")
img = np.array(img)
print(img.shape)
img = img.astype("int32")
print(img.shape)
print(img[0,0])
# for i in range(2048):
#     for j in range(2048):
#         img[i,j] = np.int(img[i,j])
# print(img[0,0])
for i in range(2048):
    for j in range(2048):
        if img[i,j] == 1:
            img[i,j] = 255
# print(img)
img = img[:, :, np.newaxis]
img = img.repeat([3], axis=2)

for i in coordinate:
    print(img[i[0],i[1], 0], img[i[0],i[1], 1], img[i[0],i[1], 2])
    img[i[1],i[0], 0] = 0
    img[i[1],i[0], 1] = 255
    img[i[1],i[0], 2] = 0
    img[i[1]-1,i[0], 0] = 0
    img[i[1]-1,i[0], 1] = 255
    img[i[1]-1,i[0], 2] = 0
    img[i[1],i[0]-1, 0] = 0
    img[i[1],i[0]-1, 1] = 255
    img[i[1],i[0]-1, 2] = 0
    img[i[1]+1,i[0], 0] = 0
    img[i[1]+1,i[0], 1] = 255
    img[i[1]+1,i[0], 2] = 0
    img[i[1],i[0]+1, 0] = 0
    img[i[1],i[0]+1, 1] = 255
    img[i[1],i[0]+1, 2] = 0
print(img.shape)

cv2.imwrite("hotspot/test9.png", img)