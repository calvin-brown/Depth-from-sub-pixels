# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:55:49 2020

@author: Calvin Brown
"""

import numpy as np
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

green1 = bayer[::2, 1::2]
green2 = bayer[1::2, ::2]

plt.figure()
plt.imshow(green1 - green2)