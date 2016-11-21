import cv2
import numpy as np
from collections import defaultdict
import os

class pxl:
  def __init__(self,x=0,y=0):
    self.x = x
    self.y = y

  def __eq__(self, other):
    return self.x == other.x and self.y == other.y

  def __hash__(self):
    return 1

palette = defaultdict(int)

def threshold(x):
    return palette[x]

def fn_log_scale(img_ft, img_scle):
      img_ft_scle = np.zeros((img_ft.shape[0], img_ft.shape[1]), dtype=np.float32)
      max_val = 0.0
      for row in range(0, img_ft.shape[0]):
          for col in range(0, img_ft.shape[1]):
                img_ft_scle[row][col] = np.sqrt(img_ft[row][col].real**2 + img_ft[row][col].imag**2)
                if max_val < img_ft_scle[row][col]:
                    max_val = img_ft_scle[row][col]
      for row in range(0, img_ft.shape[0]):
          for col in range(0, img_ft.shape[1]):
                img_scle[row][col] = (255*img_ft_scle[row][col])/(max_val-0)

if __name__ == "__main__":
  img_o = cv2.imread('chess1.png')
  img_org = np.zeros((img_o.shape[0], img_o.shape[1]), dtype=np.float32)

  rid = 1
  rbody = defaultdict(set)
  rbound = defaultdict(set)
  radj = defaultdict(set)

  for i in range(0,img_o.shape[0]):
    for j in range(0,img_o.shape[1]):
      img_org[i][j] = np.sqrt(img_o[i][j][0]**2 + img_o[i][j][1]**2 + img_o[i][j][2]**2)

  sigma = 20
  t1 = 0.5
  sigma2 = 40
  t2 = 0.1
  sigma3 = 145
  t3 = 0.1
  #print "here 1"
  img_sup = np.zeros((img_org.shape[0]*2+1, img_org.shape[1]*2+1), dtype=np.int32)
  for i in range(0,img_org.shape[0]-1):
    for j in range(0,img_org.shape[1]-1):
      img_sup[i*2+1][j*2+1+1] = abs(img_org[i][j+1] - img_org[i][j])
      img_sup[i*2+1+1][j*2+1] = abs(img_org[i+1][j] - img_org[i][j])

  for i in range(0,img_org.shape[0]):
    for j in range(0,img_org.shape[1]):   
      #print "here 2"   
      img_sup[i*2+1][j*2+1] = rid
      rbody[rid].add(pxl(i*2+1,j*2+1))
      rbound[rid].add(pxl(i*2+1+1,j*2+1))
      rbound[rid].add(pxl(i*2+1,j*2+1+1))
      rbound[rid].add(pxl(i*2+1-1,j*2+1))
      rbound[rid].add(pxl(i*2+1,j*2+1-1))
      rid = rid + 1
  #print img_sup

  for i in range(1,img_sup.shape[0]-2,2):
    for j in range(1,img_sup.shape[1]-2,2):
      #print img_sup[i][j]
      #print "here progress..."
      u = set.intersection(rbound[img_sup[i][j]],rbound[img_sup[i][j+2]])
      if len(u) > 0:
        radj[img_sup[i][j]].add(img_sup[i][j+2])
      u = set.intersection(rbound[img_sup[i][j]],rbound[img_sup[i+2][j]])
      if len(u) > 0:
        radj[img_sup[i][j]].add(img_sup[i+2][j])

  for i in range(1,img_sup.shape[0]-2,2):
    u = set.intersection(rbound[img_sup[i][img_sup.shape[1]-2]],rbound[img_sup[i+2][img_sup.shape[1]-2]])
    if len(u) > 0:
      radj[img_sup[i][img_sup.shape[1]-2]].add(img_sup[i+2][img_sup.shape[1]-2])

  for j in range(1,img_sup.shape[1]-2,2):
    u = set.intersection(rbound[img_sup[img_sup.shape[0]-2][j]],rbound[img_sup[img_sup.shape[0]-2][j+2]])
    if len(u) > 0:
      radj[img_sup[img_sup.shape[0]-2][j]].add(img_sup[img_sup.shape[0]-2][j+2])

  #for i in range(1,rid):
  #  for j in range(i+1,rid):
  #    #os.system('clear')
  #    #print rid, i, j
  #    u = set.intersection(rbound[i],rbound[j])
  #    if len(u) > 0:
  #      radj[i].add(j)
  radj[rid-1] = set()

  #for i in radj:
  #  print "index", i
  #  for j in radj[i]:
  #    print j
  #  print "--"
  
  flag = 1
  iteration = 0
  print img_sup
  
  #Phagocyte 1
  while (flag == 1):
    flag = 0
    #print "iteration-",iteration
    iteration = iteration + 1
    for index in radj:
      #print index, ":",
      u = set.copy(radj[index])
      if len(u) > 0:
        for i in u:
          #print i, "-",
          W = 0
          eliminate_bound = set.intersection(rbound[index],rbound[i])
          #print len(eliminate_bound),",",
          for iterate_bound in eliminate_bound:
            if img_sup[iterate_bound.x,iterate_bound.y] <= sigma:
              W = W + 1
          if len(eliminate_bound) > 0 and float(W)/len(eliminate_bound) >= t1:
            flag = 1        
            radj[index] = set.union(radj[index],radj[i])
            radj[i] = set()
            for j in rbody[i]:
              img_sup[j.x,j.y] = index
            rbody[index] = set.union(rbody[index],rbody[i])
            rbody[i] = set()          
            rbound[index] = set.union(rbound[index],rbound[i])
            rbound[i] = set()
            for iterate_bound in eliminate_bound:
              rbound[index].discard(iterate_bound)
            radj[index].discard(i)
      #print " "

  print img_sup[1::2,1::2]


  flag = 1
  iteration = 0      
  #Phagocyte 2
  while (flag == 1):
    flag = 0
    #print "iteration-",iteration
    iteration = iteration + 1
    for index in radj:
      #print index, ":",
      u = set.copy(radj[index])
      if len(u) > 0:
        for i in u:
          #print i, "-",
          W = 0
          eliminate_bound = set.intersection(rbound[index],rbound[i])
          #print len(eliminate_bound),
          for iterate_bound in eliminate_bound:
            if img_sup[iterate_bound.x,iterate_bound.y] <= sigma2:
              W = W + 1
          #print W,len(rbound[index]),len(rbound[i]),",",
          if len(eliminate_bound) > 0 and float(W)/min(len(rbound[index]),len(rbound[i])) >= t2:
            #print ".",
            flag = 1        
            radj[index] = set.union(radj[index],radj[i])
            radj[i] = set()
            for j in rbody[i]:
              img_sup[j.x,j.y] = index
            rbody[index] = set.union(rbody[index],rbody[i])
            rbody[i] = set()          
            rbound[index] = set.union(rbound[index],rbound[i])
            rbound[i] = set()
            for iterate_bound in eliminate_bound:
              rbound[index].discard(iterate_bound)
            radj[index].discard(i)
      #print " "


  flag = 1
  iteration = 0      
  #Weakness
  while (flag == 1):
    flag = 0
    #print "iteration-",iteration
    iteration = iteration + 1
    for index in radj:
      #print index, ":",
      u = set.copy(radj[index])
      if len(u) > 0:
        for i in u:
          #print i, "-",
          W = 0
          eliminate_bound = set.intersection(rbound[index],rbound[i])
          #print len(eliminate_bound),
          for iterate_bound in eliminate_bound:
            if img_sup[iterate_bound.x,iterate_bound.y] <= sigma3:
              W = W + 1
          #print W,len(rbound[index]),len(rbound[i]),",",
          if len(eliminate_bound) > 0 and float(W)/len(eliminate_bound) >= t3:
            #print ".",
            flag = 1        
            radj[index] = set.union(radj[index],radj[i])
            radj[i] = set()
            for j in rbody[i]:
              img_sup[j.x,j.y] = index
            rbody[index] = set.union(rbody[index],rbody[i])
            rbody[i] = set()          
            rbound[index] = set.union(rbound[index],rbound[i])
            rbound[i] = set()
            for iterate_bound in eliminate_bound:
              rbound[index].discard(iterate_bound)
            radj[index].discard(i)
      #print " "

  img_disp = np.zeros((img_o.shape[0], img_o.shape[1]), dtype=np.uint8)
  
  regions_cnt = 0

  for i in radj:
    if len(radj[i]) > 0:
      regions_cnt = regions_cnt + 1

  start = 255/5
  val = start

  for i in radj:
    if len(radj[i]) > 0:
      palette[i] = val
      val = val + start
      if val > 255:
        val = start

  for i in rbound:
    for j in rbound[i]:
      img_o[(j.x-1)/2][(j.y-1)/2] = [0,255,0]

  img_disp = np.asarray(map(threshold,img_sup[1::2,1::2].flatten()))
  img_disp = img_disp.reshape(img_o.shape[0], img_o.shape[1])
  #temp = temp.resize(img_o.shape[0], img_o.shape[1])
  #img_disp = temp

  temp = img_sup[1::2,1::2]

  print img_sup[1::2,1::2]
  print img_disp



  img_scle = np.zeros((temp.shape[0], temp.shape[1]), dtype=np.uint8)
  fn_log_scale(temp, img_scle)

  cv2.imshow("intensity",img_o)
  #cv2.waitKey(0)

  #cv2.imshow("segment",img_disp)
  cv2.waitKey(0)