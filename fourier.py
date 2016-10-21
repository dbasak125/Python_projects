import cv2
import numpy as np

def fn_ft(img_org, img_ft):
	img_ft_intr = np.zeros((img_org.shape[0], img_org.shape[1]), dtype=complex)
    	for row in range(0, img_org.shape[0]):
        	for col in range(0, img_org.shape[1]):
            		for n in range(0, img_org.shape[1]):
                		img_ft_intr[row][col] = img_ft_intr[row][col] + ((img_org[row][n]*np.exp(((-2j)*np.pi*n*col)/img_org.shape[1]))/img_org.shape[1])
	for col in range(0, img_org.shape[1]):
	        for row in range(0, img_org.shape[0]):
        	    	for m in range(0, img_org.shape[0]):
                		img_ft[row][col] = img_ft[row][col] + ((img_ft_intr[m][col]*np.exp(((-2j)*np.pi*m*row)/img_ft_intr.shape[0]))/img_ft_intr.shape[0])

def fn_ft_scale(img_ft, img_scle):
    	img_ft_scle = np.zeros((img_ft.shape[0], img_ft.shape[1]), dtype=np.float32)
    	max_val = 0.0
    	for row in range(0, img_ft.shape[0]):
        	for col in range(0, img_ft.shape[1]):
            		img_ft_scle[row][col] = np.sqrt(img_ft[row][col].real**2 + img_ft[row][col].imag**2)
            		if max_val < img_ft_scle[row][col]:
                		max_val = img_ft_scle[row][col]
    	c = 255/np.log(1+max_val)
    	for row in range(0, img_ft.shape[0]):
        	for col in range(0, img_ft.shape[1]):
            		img_scle[row][col] = np.int(c * np.log(1+img_ft_scle[row][col]))

def fn_ft_shift(img_scle):
	img_scl_temp = np.copy(img_scle)
	j = img_scl_temp.shape[0]
	k = img_scl_temp.shape[1]
	A = img_scl_temp[0:(j/2)-1,0:(k/2)-1]
	B = img_scl_temp[0:(j/2)-1,(k/2):k]
	C = img_scl_temp[(j/2):j,0:(k/2)-1]
	D = img_scl_temp[(j/2):j,(k/2):k]
	E = np.vstack((D,B))
	F = np.vstack((C,A))
	img_scle = np.hstack((E,F))
	cv2.imwrite('img_scaled_ft.png', img_scle)

def fn_ift(img_ft, img_ift):
	img_ft_intr = np.zeros((img_ft.shape[0], img_ft.shape[1]), dtype=complex)
        for row in range(0, img_ft.shape[0]):
                for col in range(0, img_ft.shape[1]):
                        for n in range(0, img_ft.shape[1]):
                                img_ft_intr[row][col] = img_ft_intr[row][col] + (img_ft[row][n]*np.exp(((2j)*np.pi*n*col)/img_ft.shape[1]))
        for col in range(0, img_ft.shape[1]):
                for row in range(0, img_ft.shape[0]):
                        for m in range(0, img_ft.shape[0]):
                                img_ift[row][col] = img_ift[row][col] + (img_ft_intr[m][col]*np.exp(((2j)*np.pi*m*row)/img_ft_intr.shape[0]))
			img_ift[row][col] = img_ift[row][col].real

img_org = cv2.imread('gpA_0.png',cv2.IMREAD_GRAYSCALE)
img_ft = np.zeros((img_org.shape[0], img_org.shape[1]), dtype=complex)
img_scle = np.zeros((img_org.shape[0], img_org.shape[1]), dtype=np.uint8)
img_ift = np.zeros((img_org.shape[0], img_org.shape[1]), dtype=complex)

fn_ft(img_org, img_ft)
fn_ft_scale(img_ft, img_scle)
fn_ft_shift(img_scle)
fn_ift(img_ft, img_ift)
img_ift = img_ift.astype(np.uint8)
cv2.imwrite('img_reconst.png',img_ift)

rl = 0.0
re = 0.0
for i in range(0, img_ift.shape[0]):
	for j in range(0, img_ift.shape[1]):
		re = re + (img_org[i][j] - img_ift[i][j])**2
	rl = rl + (re/img_ift.size)
	re = 0

print 'MSE: '+str(rl)


