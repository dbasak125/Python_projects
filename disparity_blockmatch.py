# Disparity map generation using Block-matching method (metric: SSD)
#
# Debaditya Basak
# 12/15/2016
#
import numpy as np
import cv2
from matplotlib import pyplot as plt

im1 = cv2.imread('view1.png',cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('view3.png',cv2.IMREAD_GRAYSCALE)
ref = cv2.imread('disp1.png',cv2.IMREAD_GRAYSCALE)
row = 0
box = 7  #-----------> this variable for block-size
exc = box/2
disp_limit = 40

#padding the image views
a = np.zeros((im1.shape[0]+(box/2)+(box/2), im1.shape[1]+(box/2)+(box/2)), dtype=np.uint8)
b = np.zeros((im2.shape[0]+(box/2)+(box/2), im2.shape[1]+(box/2)+(box/2)), dtype=np.uint8)
a[box/2:a.shape[0]-box/2:1,box/2:a.shape[1]-box/2:1] = im1[::1,::1]
b[box/2:b.shape[0]-box/2:1,box/2:b.shape[1]-box/2:1] = im2[::1,::1]

def measure_l2r(i,j): 
  return np.power((a[row:row+box:1,i:i+box:1]-b[row:row+box:1,j:j+box:1]),2).sum()

if __name__ == "__main__":
  Sl2r = np.zeros((im1.shape[0], im1.shape[1]), dtype=np.uint8)
  while row < im1.shape[0]:
    print "processed -> row-",row,"of",im1.shape[0]
    
    #left-2-right scan
    for i in range(0, im1.shape[1]):
      ssd = float('inf')
      disp = 0
      for j in range(i-disp_limit, i+1):
        if j<0 or j>im1.shape[1]-1:
          continue
        else:
          tmp = measure_l2r(i,j)
          if tmp < ssd:
            ssd = tmp
            disp = abs(i-j)

      Sl2r[row][i] = disp
      
    row = row + 1

  rl = 0.0
  re = 0.0
  re = np.power((ref[0:im1.shape[0]-1:1,0:im1.shape[1]-1:1]-Sl2r[0:im1.shape[0]-1:1,0:im1.shape[1]-1:1]),2).sum()
  rl = re/Sl2r.size

  print 'MSE: '+str(rl)
  cv2.imwrite('blockmatch_res.png',Sl2r)

  mod = cv2.equalizeHist(Sl2r)
  cv2.imwrite('blockmatch_res_mod.png',mod)
  plt.imshow(mod,'gray')
  plt.show()
