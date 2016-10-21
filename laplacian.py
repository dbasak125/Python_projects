import numpy as np
import cv2

def pad(img_org,img_pad):
	#img_pad = np.zeros((img_org.shape[0]+2,img_org.shape[1]+2), dtype=np.uint8)
	for i in range(1,img_org.shape[0]+1):
		for j in range(1,img_org.shape[1]+1):
			img_pad[i][j] = img_org[i-1][j-1]

def blur(img,img_pad,gauss):
	for i in range(1,img.shape[0]+1):
                for j in range(1,img.shape[1]+1):
			img[i-1][j-1] = 0
                        for k in range(0,3):
				for l in range(0,3):
					img[i-1][j-1] = img[i-1][j-1] + (gauss[k][l]*img_pad[i-1+k][j-1+l])

def downscale(img_cp,img):
	#img = np.reshape(img, (img_cp.shape[0]/2,img_cp.shape[1]/2))
	for i in xrange(0,img_cp.shape[0]-1,2):
		for j in xrange(0,img_cp.shape[1]-1,2):
			#print str(i) + '  ' + str(j)
			#print img.shape
			#print img_cp.shape
			img[i/2][j/2] = img_cp[i][j]	

def upscale(A, img_up):
	for i in range(0,(A.shape[0]*2)):
		for j in range(0,(A.shape[1]*2)):
			img_up[i][j] = A[i/2][j/2]

list = []
list2 = []
gauss = np.array((0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625))
gauss = np.reshape(gauss, (3,3))
#print gauss
img_org = cv2.imread('duck_gs.png',cv2.IMREAD_GRAYSCALE)
img = np.copy(img_org)
img = img.astype(np.float32)
for i in range(0,6):
	img_pad = np.zeros((img.shape[0]+2,img.shape[1]+2), dtype=np.float32)
	pad(img,img_pad)
	img_o = np.copy(img)
	#print img_pad
	blur(img,img_pad,gauss)
	A = img_o
	B = img
	if img_o.shape[0]<img.shape[0]:
		B = img[0:img.shape[0]-1,:]
	if img.shape[0]<img_o.shape[0]:
		A = img_o[0:img_o.shape[0]-1,:]
	if img_o.shape[1]<img.shape[1]:
		B = B[:,0:img.shape[1]-1]
	if img.shape[1]<img_o.shape[1]:
		A = A[:,0,img_o.shape[1]-1]
	#print img
	#list.append(B)
	img_f = img.astype(np.uint8)
        cv2.imwrite('gaus_'+str(i)+'.png', img_f)
	list2.append(A-B)
	img_g = list2[i].astype(np.uint8)
	cv2.imwrite('lapl_'+str(i)+'.png', img_g)
	img_cp = np.copy(img)
	img = np.zeros((img_cp.shape[0]/2,img_cp.shape[1]/2), dtype=np.float32)
	downscale(img_cp,img)
	list.append(img)

for i in range(0,5):
	img_up = np.zeros((list[4-i].shape[0]*2,list[4-i].shape[1]*2), dtype=np.float32)
	upscale(list[4-i],img_up)
	A = img_up
        B = list2[4-i]
        if A.shape[0]<B.shape[0]:
                B = B[0:B.shape[0]-1,:]
        if B.shape[0]<A.shape[0]:
                A = A[0:A.shape[0]-1,:]
        if A.shape[1]<B.shape[1]:
                B = B[:,0:B.shape[1]-1]
        if B.shape[1]<A.shape[1]:
                A = A[:,0,A.shape[1]-1]
	img_r = A + B
	img_r = img_r.astype(np.uint8)
	cv2.imwrite('lt_recon_'+str(4-i)+'.png', img_r)

rl = 0.0
re = 0.0
for i in range(0, list2[0].shape[0]):
	for j in range(0, list2[0].shape[1]):
		re = re + (img_org[i][j] - list2[0][i][j])**2
	rl = rl + (re/list2[0].size)
	re = 0

print 'MSE: '+str(rl)
