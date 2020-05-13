from PIL import Image
import skimage
from skimage import data, io, filters,morphology,segmentation,color
from scipy import ndimage
from skimage import color
import numpy as np
import math
from math import sqrt
import sys, getopt

def func(t):
    if (t > 0.008856):
        return np.power(t, 1/3.0);
    else:
        return 7.787 * t + 16 / 116.0;

def rgb2lab ( inputColor ) :

    #Conversion Matrix
    matrix = [[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]]

    # RGB values lie between 0 to 1.0
    rgb = [1.0, 0, 0] # RGB

    cie = np.dot(matrix, rgb);

    cie[0] = cie[0] /0.950456;
    cie[2] = cie[2] /1.088754; 

    # Calculate the L
    L = 116 * np.power(cie[1], 1/3.0) - 16.0 if cie[1] > 0.008856 else 903.3 * cie[1];

    # Calculate the a 
    a = 500*(func(cie[0]) - func(cie[1]));

    # Calculate the b
    b = 200*(func(cie[1]) - func(cie[2]));

    #  Values lie between -128 < b <= 127, -128 < a <= 127, 0 <= L <= 100 
    Lab = [b , a, L]; 

    # OpenCV Format
    L = L * 255 / 100;
    a = a + 128;
    b = b + 128;
    Lab_OpenCV = [b , a, L]; 

def params(c1,c2):
    L1 = c1[0]
    L2 = c2[0]
    a1 = c1[1]
    a2 = c2[1]
    b1 = c1[2]
    b2 = c2[2]
    delta_L = L1 - L2
    med_C = (sqrt(a1*a1 + b1*b1) + sqrt(a2*a2 + b2*b2))/2
    med_L = (L1+L2)/2
    a1_sht = a1 + a1*(1 - sqrt(med_C**(7)/(med_C**(7)+25**(7))))/2
    a2_sht = a2 + a1*(2 - sqrt(med_C**(7)/(med_C**(7)+25**(7))))/2
    delta_C = sqrt(a2_sht*a2_sht + b2*b2) - sqrt(a1_sht*a1_sht + b1*b1)
    h1_sht = math.atan2(b1,a1_sht) % 360
    h2_sht = math.atan2(b2,a2_sht) % 360
    
    if abs(h1_sht-h2_sht) <= 180:
        delta_h_sht = h2_sht - h1_sht
        H_sht = (h2_sht + h1_sht)/2
    elif (abs(h1_sht-h2_sht) > 180) and (h2_sht <= h1_sht):
        delta_h_sht = h2_sht - h1_sht + 360
        H_sht = (h2_sht + h1_sht + 360)/2
    elif (abs(h1_sht-h2_sht) > 180) and (h2_sht > h1_sht):
        delta_h_sht = h2_sht - h1_sht - 360
        H_sht = (h2_sht + h1_sht - 360)/2

    delta_H = 2*sqrt(sqrt(a2_sht*a2_sht + b2*b2)*sqrt(a1_sht*a1_sht + b1*b1))*np.sin(delta_h_sht/2)
    T  = 1 - 0.17*np.cos(H_sht - 30) + 0.24*np.cos(2*H_sht) + 0.32*np.cos(3*H_sht + 6) - 0.2*np.cos(4*H_sht - 63)
    SL = 1 + 0.015*(med_L-50)*(med_L-50)/sqrt(20 + (med_L-50)*(med_L-50))
    vspom_C = (sqrt(a2_sht*a2_sht + b2*b2) + sqrt(a1_sht*a1_sht + b1*b1))/2
    SC = 1 + 0.045*vspom_C
    SH = 1 + 0.015*vspom_C*T
    RT = -2*sqrt((vspom_C**(7)/(vspom_C**(7)+25**(7))))*np.sin(60*math.exp(-((H_sht-275)/25)**2))
    
    E00 = sqrt((delta_L/SL)*(delta_L/SL) + (delta_C/SC)*(delta_C/SC) + (delta_H/SH)*(delta_H/SH) + RT*delta_C*delta_C/(SC*SH))
    E00CH = sqrt((delta_C/SC)*(delta_C/SC) + (delta_H/SH)*(delta_H/SH) + RT*delta_C*delta_C/(SC*SH))
    return E00, E00CH

def conds(pix,tau, bg):
    E00, E00CH = params(pix, bg)
    cond1 = (E00 < tau) or((E00CH < tau) and (pix[0]>bg[0]))
    cond2 = sqrt(pix[1]*pix[1]+pix[2]*pix[2]) < 2.304
    return cond1, cond2
    
def method(fd):
    gaussed = skimage.filters.gaussian(fd, 0.5)
    lim = skimage.color.rgb2lab(fd)
    lim = (lim + [0, 128, 128]) / [100, 255, 255]
    #return lim
    print(lim[100,100,1])
    hist = np.zeros((26,21,21))
    amount_bins = 0
    for i in range(fd.shape[0]):
        for j in range(fd.shape[1]):
            L = lim[i,j,0]#%100  
            a = lim[i,j,1]#%256 - 128
            b = lim[i,j,2]#%256 - 128
            Cp = math.sqrt(a*a+b*b)
            if (Cp < 4.075) and (74<L<101) and(-11<b<11) and (-11<a<11):       
                hist[int(L)-75,int(a)-10,int(b)-10] +=1
                if hist[int(L)-75,int(a)-10,int(b)-10] == 1:
                    amount_bins+=1

    a = round(0.005*amount_bins)
    maxes = np.zeros(a)
    indc = []
    temp = []
    for i in range(26):
        for j in range(21):
            for k in range(21):
                temp.append((hist[i,j,k],i+75,j-10,k-10))
    temp.sort()
    maxes = temp[len(temp):len(temp)-a-1:-1]
    count = 0
    for i in range(a):
        count += maxes[i][0]
    print(maxes)
    l = 0
    at = 0
    bt = 0
    for i in range(a):
        wi = maxes[i][0]/count
        l += maxes[i][1]*wi
        at += maxes[i][2]*wi
        bt += maxes[i][3]*wi

    newobj = np.zeros((300,300,3))
    bg = [l,at,bt]
    '''for i in range(300):
        for j in range(300):
            newobj[i,j,0] = l
            newobj[i,j,1] = at
            newobj[i,j,2] = bt
    
    #return (newobj - [0, 128, 128]) *[100, 255, 255]
    return newobj'''
    MCore = np.zeros((fd.shape[0],fd.shape[1]))
    MBase = np.zeros((fd.shape[0],fd.shape[1]))
    for i in range(fd.shape[0]):
        for j in range(fd.shape[1]):
            pix = lim[i,j]
            cond1, cond2 = conds(pix, 3.047, bg)
            cond1b, cond2b = conds(pix, 7.049, bg)
            if cond1 or cond2:
                MCore[i,j] = 1
            else:
                MCore[i,j] = 0
            if cond1b or cond2b:
                MBase[i,j] = 1
            else:
                MBase[i,j] = 0
                
    
    MConstr = np.zeros((fd.shape[0],fd.shape[1]))
    for i in range(fd.shape[0]):
        for j in range(fd.shape[1]):
            pix = MCore[i,j]
            if neighb(pix, MCore, 10, i, j) > 0.02:
                MConstr[i,j] = 1
            else:
                MConstr[i,j] = 0
    MCore = skimage.morphology.dilation(MCore,skimage.morphology.disk(1))
    for i in range(fd.shape[0]):
        for j in range(fd.shape[1]):
            if (MCore[i,j]>0) or (MConstr[i,j]>0):
                MConstr[i,j] = 1
    M = np.zeros((fd.shape[0],fd.shape[1]))
    for i in range(fd.shape[0]):
        for j in range(fd.shape[1]):
            if (MCore[i,j]>0) or (MConstr[i,j]>0 and MBase[i,j]>0):
                M[i,j] = 1
    MBG = skimage.morphology.reconstruction(MCore, M)
    for i in range(fd.shape[0]):
        for j in range(fd.shape[1]):
            if MBG[i,j]>0:
                MBG[i,j] = 1

    lim = skimage.filters.gaussian(lim, 5) - skimage.filters.gaussian(mult(lim, MBG),5)
    lim = divide(lim, skimage.filters.gaussian(M, 5))

    M = np.zeros((fd.shape[0],fd.shape[1]))
    for i in range(fd.shape[0]):
        for j in range(fd.shape[1]):
            pix = lim[i,j]
            if (MBG[i,j]>0) or (cond3(pix,bg)):
                M[i,j] = 1
            
    
    M = skimage.morphology.reconstruction(MBG, M)
    MBG = skimage.morphology.dilation(MBG,skimage.morphology.disk(5))
    for i in range(fd.shape[0]):
        for j in range(fd.shape[1]):
            if (MBG[i,j]>0) and (M[i,j]>0):
                MBG[i,j] = 255
            else:
                MBG[i,j] = 0
    return MBG

def cond3(p,bg):
    return (p[0]>(bg[0]-1.32*2.30)) and (sqrt(p[1]*p[1]+p[2]*p[2]) < 3.06*2.30)
    
def mult(A, M):
    M = inv(M)
    newobj = np.zeros(fd.shape)
    for i in range(fd.shape[0]):
        for j in range(fd.shape[1]):
            newobj[i,j] = A[i,j]*M[i,j]
    return newobj

def divide(A,M):
    newobj = np.zeros(fd.shape)
    for i in range(fd.shape[0]):
        for j in range(fd.shape[1]):
            newobj[i,j] = A[i,j]/M[i,j]
    return newobj

def inv(M):
    newobj = np.zeros((fd.shape[0],fd.shape[1]))
    for i in range(fd.shape[0]):
        for j in range(fd.shape[1]):
            newobj[i,j] = 255-M[i,j]
    return newobj

def neighb(pix, M, n, i1, j1):
    k1 = 0
    k = 1
    if (i1 >= 10) and (abs(fd.shape[0]-i1) >= 10) and (j1 >= 10) and (abs(fd.shape[1]-j1) >= 10):
        k1 = 0
        k = 0
        for i in range(i1-10,i1+10):
            for j in range(j1-10,j1+10):
                if M[i,j] == 1:
                    k1 += 1
                k += 1
    elif (i1 < 10):
        if (j1 < 10):
            k1 = 0
            k = 0
            for i in range(i1+10):
                for j in range(j1+10):
                    if M[i,j] == 1:
                        k1 += 1
                    k += 1
        elif (abs(fd.shape[1]-j1) < 10):
            k1 = 0
            k = 0
            for i in range(i1+10):
                for j in range(j1-10,fd.shape[1]):
                    if M[i,j] == 1:
                        k1 += 1
                    k += 1
        else:
            k1 = 0
            k = 0
            for i in range(i1+10):
                for j in range(j1-10,j1+10):
                    if M[i,j] == 1:
                        k1 += 1
                    k += 1
    elif (abs(fd.shape[0]-i1) < 10):
        if (j1 < 10):
            k1 = 0
            k = 0
            for i in range(i1-10,fd.shape[0]):
                for j in range(j1+10):
                    if M[i,j] == 1:
                        k1 += 1
                    k += 1
        if (abs(fd.shape[1]-j1) < 10):
            k1 = 0
            k = 0
            for i in range(i1-10,fd.shape[0]):
                for j in range(j1-10,fd.shape[1]):
                    if M[i,j] == 1:
                        k1 += 1
                    k += 1
        else:
            k1 = 0
            k = 0
            for i in range(i1-10,fd.shape[0]):
                for j in range(j1-10,j1+10):
                    if M[i,j] == 1:
                        k1 += 1
                    k += 1
    return k1/k
            
  
                
                
im = Image.open("/Users/vladimirlisovoi/desktop/testim/s1_test_01.png")
data = np.array(im)
fd = data.astype(np.float)
fd = method(fd)
#fd = skimage.color.lab2rgb(fd)
fd = inv(fd)
fd = fd.astype(np.uint8) 
omg = Image.fromarray(fd)
omg.show()
omg.save("/Users/vladimirlisovoi/desktop/testim/out.png")
