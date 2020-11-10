# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:55:49 2020

@author: Calvin Brown
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


r = 3040
c = 4056

with open('image.jpg', 'rb') as f:
    data = f.read()
data = list(data)

img = data[-18711040 + 2**15:]
img = np.reshape(img, (r + 16, -1))
img = img[:r, :int(c * 12/8)].astype('uint16')

bayer = np.zeros((r, c), dtype='uint16')
bayer[:, ::2] = (img[:, ::3] << 4) + (img[:, 2::3] & 15)
bayer[:, 1::2] = (img[:, 1::3] << 4) + (img[:, 2::3] >> 4)

# red_img = bayer[1::2, 1::2]
# green_img = (bayer[::2, 1::2] + bayer[1::2, ::2]) / 2
# blue_img = bayer[::2, ::2]

# for single_color in [red_img, green_img, blue_img]:
#     plt.figure()
#     plt.imshow(single_color, vmin=0, vmax=2000)
    
# plt.figure()
# plt.imshow(np.stack((red_img, green_img, blue_img), axis=-1) / 2000)

# green1 = bayer[::2, 1::2]
# green2 = bayer[1::2, ::2] # green2 is shifted "southwest" of green1
green1 = bayer[::4, 1::4]
green2 = bayer[::4, 3::4]

plt.figure()
plt.imshow(green1 - green2)

# convert images to 8-bit for openCV
green1_8bit = np.zeros(green1.shape, dtype='uint8')
green1_8bit = cv.normalize(green1, green1_8bit, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
green1_8bit = green1_8bit.astype('uint8')
green2_8bit = np.zeros(green2.shape, dtype='uint8')
green2_8bit = cv.normalize(green2, green2_8bit, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
green2_8bit = green2_8bit.astype('uint8')

blocks = [11, 21, 31, 41]

for i in range(len(blocks)):
    
    # compute depth (disparity) image
    window_size=5
    # stereo = cv.StereoSGBM_create(minDisparity=0,
    #                               numDisparities=16,
    #                               blockSize=16,
    #                               P1=8*3*window_size**2,
    #                               P2=32*3*window_size**2,
    #                               disp12MaxDiff=1,
    #                               uniquenessRatio=10,
    #                               speckleWindowSize=100,
    #                               speckleRange=32
    #                               )
    # stereo = cv.StereoSGBM_create(minDisparity=0,
    #                               numDisparities=16,
    #                               blockSize=16,
    #                               P1=8*3*window_size**2,
    #                               P2=32*3*window_size**2,
    #                               disp12MaxDiff=1,
    #                               uniquenessRatio=10,
    #                               speckleWindowSize=100,
    #                               speckleRange=32,
    #                               textureThreshold=10)
    stereo = cv.StereoSGBM_create(minDisparity=-1,
                                  numDisparities=64,
                                  blockSize=5,
                                  uniquenessRatio=5,
                                  speckleWindowSize=5,
                                  speckleRange=5,
                                  disp12MaxDiff=1,
                                  P1=8*3*window_size**2,
                                  P2=32*3*window_size**2)
    # stereo = cv.StereoBM_create(numDisparities=16, blockSize=blocks[i])
    disparity = stereo.compute(green1_8bit, green2_8bit)
    
    plt.figure()
    # plt.imshow(disparity, 'gray')
    plt.imshow(disparity)
    plt.colorbar()