# Disparity map generation using Dynamic-programming method (metric: SAD)
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
row = 1

if __name__ == "__main__":
  S = np.zeros((im1.shape[0], im1.shape[1]), dtype=np.uint8)
  C = np.zeros((im1.shape[1]-2, im1.shape[1]-2), dtype=np.float32)
  E = np.zeros((im1.shape[1]-2, im1.shape[1]-2), dtype=np.float32)
    
  while row < im1.shape[0]-1:
    print "processed -> row-",row,"of",im1.shape[0]

    R = np.zeros((im1.shape[1]-2, im1.shape[1]-2), dtype=np.uint8)
    
    occlusion = 20
    N = im1.shape[1]-2
    M = im1.shape[1]-2

    for i in range(1, N):
      C[i][0] = i * occlusion
      C[0][i] = i * occlusion

    for i in range(1, N):
      for j in range(1, M):
        min1 = C[i-1][j-1] + abs(im1[row][i]-im2[row][j])
        min2 = C[i-1][j] + occlusion
        min3 = C[i][j-1] + occlusion
        C[i][j] = np.amin([min1,min2,min3])
        cmin = C[i][j]
        if cmin == min1:
          E[i][j] = 1
        elif cmin == min2:
          E[i][j] = 2
        elif cmin == min3:
          E[i][j] = 3

    p = N-1
    q = M-1
    while p!=0 and q!=0:
      R[p][q] = 255
      S[row][p] = abs(p-q)
      if E[p][q] == 1:
        p = p - 1
        q = q - 1
      elif E[p][q] == 2:
        p = p - 1
      elif E[p][q] == 3:
        q = q - 1

    row = row + 1

  rl = 0.0
  re = 0.0
  re = np.power((ref[0:im1.shape[0]-1:1,0:im1.shape[1]-1:1]-S[0:im1.shape[0]-1:1,0:im1.shape[1]-1:1]),2).sum()
  rl = re/S.size

  print 'MSE: '+str(rl)
  cv2.imwrite('dynamic_res.png',S)

  mod = cv2.equalizeHist(S)
  cv2.imwrite('dynamic_res_mod.png',mod)
  plt.imshow(mod,'gray')
  plt.show()
