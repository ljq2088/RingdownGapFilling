import numpy as np
import matplotlib.pyplot as plt

# import waveform_zxy as wf
# import fisher as fh

#先给一些常数
M_sun = 1.99*1e30
pc = 30.8396*1e15
C = 3*1e8
cG = 6.67*1e-11
pi = np.pi

#这里直接给以秒为单位的情况
C       = 299792458.         # m/s
YEAR    = 3.15581497632e7    # sec
TSUN    = 4.92549232189886339689643862e-6 # mass of sun in seconds (G=C=1)
MPC     = 3.08568025e22/C    # mega-Parsec in seconds

#波形的参数
f1 = [0,1.5251 , 0.6000 , 1.8956 , 2.3000]
f2 = [0,-1.1568 , -0.2339 , -1.3043 , -1.5056]
f3 = [0,0.1292, 0.4175, 0.1818, 0.2244]
q1 = [0,0.7, -0.3, 0.9, 1.1929]
q2 = [0,1.4187, 2.3561, 2.3430, 3.1191]
q3 = [0,-0.4990, -0.2277, -0.4810, -0.4825]

#这个是红移到光度距离的转化，对应的单位是MPC
H0 = 67.4  #（哈勃常数） 
Ωm = 0.315 #（物质密度参数） 
ΩΛ = 0.685 #（暗能密度参数）
c0=299792.458
from scipy import integrate
import math
def E(z):
    return 1/math.sqrt(Ωm*(1+z)**3+ΩΛ)

def DL(z):
    v,err=integrate.quad(E,0,z)
    return (1+z)*c0/H0*v

#print(DL(1))

#针对天琴的，因为取的天琴的omega
########################发现这个转移频率，好像对波形的绝对值，没有什么影响，赢，那就是说联合观测波形不用管咯
wtq = 1.99*1e-5
#我猜这里的w就是S(f)中的频率
def Ib_p(w,wlm,tlm,phid ,philm = pi/3):
    #wtq 是天琴的rotation frequency
    #这是第一个22模式
    #phid = phid + pi/4  #原文要求带入此数值
    #这一部分，与原文对比过，代码上是一致的
    Ib1_1 = 2*tlm[0] *wtq*(tlm[0]**2 *(w**2 + wlm[0]**2 -4*wtq**2) - 2*1j*w*tlm[0] - 1)*np.cos(2*phid) - \
        (1+1j*w*tlm[0])*(tlm[0]**2*(w**2 - wlm[0]**2 - 4*wtq**2) - 2j * w *tlm[0] - 1)*np.sin(2*phid)
    Ib2_1 = (tlm[0]**2*(w**2 - wlm[0]**2 + 4*wtq**2) - 2j*w*tlm[0] - 1)*np.sin(2*phid)+\
        4*tlm[0]*wtq*(1+1j*w*tlm[0])*np.cos(2*phid)
    Ib3_1 = (tlm[0]*(w-wlm[0]-2*wtq) - 1j)*(tlm[0]*(w + wlm[0] -2*wtq)-1j)*(tlm[0]*(w-wlm[0]+2*wtq)-1j)*(tlm[0]*(w+wlm[0]+2*wtq)-1j)
    Ib_22 = (Ib1_1 * np.cos(philm) - Ib2_1 * tlm[0] * wlm[0] *np.sin(philm))*tlm[0]/Ib3_1
    #这是第二个21模式
    Ib1_2 = 2*tlm[1] *wtq*(tlm[1]**2 *(w**2 + wlm[1]**2 -4*wtq**2) - 2*1j*w*tlm[1] - 1)*np.cos(2*phid) - \
        (1+1j*w*tlm[1])*(tlm[1]**2*(w**2 - wlm[1]**2 - 4*wtq**2) - 2j * w *tlm[1] - 1)*np.sin(2*phid)
    Ib2_2 = (tlm[1]**2*(w**2 - wlm[1]**2 + 4*wtq**2) - 2j*w*tlm[1] - 1)*np.sin(2*phid)+\
        4*tlm[1]*wtq*(1+1j*w*tlm[1])*np.cos(2*phid)
    Ib3_2 = (tlm[1]*(w-wlm[1]-2*wtq) - 1j)*(tlm[1]*(w + wlm[1] -2*wtq)-1j)*(tlm[1]*(w-wlm[1]+2*wtq)-1j)*(tlm[1]*(w+wlm[1]+2*wtq)-1j)
    Ib_21 = (Ib1_2 * np.cos(philm) - Ib2_2 * tlm[1] * wlm[1] *np.sin(philm))*tlm[1]/Ib3_2
    #这是第三个33模式
    Ib1_3 = 2*tlm[2] *wtq*(tlm[2]**2 *(w**2 + wlm[2]**2 -4*wtq**2) - 2*1j*w*tlm[2] - 1)*np.cos(2*phid) - \
        (1+1j*w*tlm[2])*(tlm[2]**2*(w**2 - wlm[2]**2 - 4*wtq**2) - 2j * w *tlm[2] - 1)*np.sin(2*phid)
    Ib2_3 = (tlm[2]**2*(w**2 - wlm[2]**2 + 4*wtq**2) - 2j*w*tlm[2] - 1)*np.sin(2*phid)+\
        4*tlm[2]*wtq*(1+1j*w*tlm[2])*np.cos(2*phid)
    Ib3_3 = (tlm[2]*(w-wlm[2]-2*wtq) - 1j)*(tlm[2]*(w + wlm[2] -2*wtq)-1j)*(tlm[2]*(w-wlm[2]+2*wtq)-1j)*(tlm[2]*(w+wlm[2]+2*wtq)-1j)
    Ib_33 = (Ib1_3 * np.cos(philm) - Ib2_3 * tlm[2] * wlm[2] *np.sin(philm))*tlm[2]/Ib3_3
    #这是第四个44模式
    Ib1_4 = 2*tlm[3] *wtq*(tlm[3]**2 *(w**2 + wlm[3]**2 -4*wtq**2) - 2*1j*w*tlm[3] - 1)*np.cos(2*phid) - \
        (1+1j*w*tlm[3])*(tlm[3]**2*(w**2 - wlm[3]**2 - 4*wtq**2) - 2j * w *tlm[3] - 1)*np.sin(2*phid)
    Ib2_4 = (tlm[3]**2*(w**2 - wlm[3]**2 + 4*wtq**2) - 2j*w*tlm[3] - 1)*np.sin(2*phid)+\
        4*tlm[3]*wtq*(1+1j*w*tlm[3])*np.cos(2*phid)
    Ib3_4 = (tlm[3]*(w-wlm[3]-2*wtq) - 1j)*(tlm[3]*(w + wlm[3] -2*wtq)-1j)*(tlm[3]*(w-wlm[3]+2*wtq)-1j)*(tlm[3]*(w+wlm[3]+2*wtq)-1j)
    Ib_44 = (Ib1_4 * np.cos(philm) - Ib2_4 * tlm[3] * wlm[3] *np.sin(philm))*tlm[3]/Ib3_4

    return Ib_22,Ib_21,Ib_33,Ib_44


#现在定义与cross缩并的Ib：
def Ib_c(w,wlm,tlm,phid , philm = pi/3):
    #wtq 是天琴的rotation frequency
    #这是第一个22模式
    #phid = phid + pi/4  #原文要求带入此数值
    philm = philm - pi/2
    #这一部分，与原文对比过，代码上是一致的
    Ib1_1 = 2*tlm[0] *wtq*(tlm[0]**2 *(w**2 + wlm[0]**2 -4*wtq**2) - 2*1j*w*tlm[0] - 1)*np.cos(2*phid) - \
        (1+1j*w*tlm[0])*(tlm[0]**2*(w**2 - wlm[0]**2 - 4*wtq**2) - 2j * w *tlm[0] - 1)*np.sin(2*phid)
    Ib2_1 = (tlm[0]**2*(w**2 - wlm[0]**2 + 4*wtq**2) - 2j*w*tlm[0] - 1)*np.sin(2*phid)+\
        4*tlm[0]*wtq*(1+1j*w*tlm[0])*np.cos(2*phid)
    Ib3_1 = (tlm[0]*(w-wlm[0]-2*wtq) - 1j)*(tlm[0]*(w + wlm[0] -2*wtq)-1j)*(tlm[0]*(w-wlm[0]+2*wtq)-1j)*(tlm[0]*(w+wlm[0]+2*wtq)-1j)
    Ib_22 = (Ib1_1 * np.cos(philm) - Ib2_1 * tlm[0] * wlm[0] *np.sin(philm))*tlm[0]/Ib3_1
    #这是第二个21模式
    Ib1_2 = 2*tlm[1] *wtq*(tlm[1]**2 *(w**2 + wlm[1]**2 -4*wtq**2) - 2*1j*w*tlm[1] - 1)*np.cos(2*phid) - \
        (1+1j*w*tlm[1])*(tlm[1]**2*(w**2 - wlm[1]**2 - 4*wtq**2) - 2j * w *tlm[1] - 1)*np.sin(2*phid)
    Ib2_2 = (tlm[1]**2*(w**2 - wlm[1]**2 + 4*wtq**2) - 2j*w*tlm[1] - 1)*np.sin(2*phid)+\
        4*tlm[1]*wtq*(1+1j*w*tlm[1])*np.cos(2*phid)
    Ib3_2 = (tlm[1]*(w-wlm[1]-2*wtq) - 1j)*(tlm[1]*(w + wlm[1] -2*wtq)-1j)*(tlm[1]*(w-wlm[1]+2*wtq)-1j)*(tlm[1]*(w+wlm[1]+2*wtq)-1j)
    Ib_21 = (Ib1_2 * np.cos(philm) - Ib2_2 * tlm[1] * wlm[1] *np.sin(philm))*tlm[1]/Ib3_2
    #这是第三个33模式
    Ib1_3 = 2*tlm[2] *wtq*(tlm[2]**2 *(w**2 + wlm[2]**2 -4*wtq**2) - 2*1j*w*tlm[2] - 1)*np.cos(2*phid) - \
        (1+1j*w*tlm[2])*(tlm[2]**2*(w**2 - wlm[2]**2 - 4*wtq**2) - 2j * w *tlm[2] - 1)*np.sin(2*phid)
    Ib2_3 = (tlm[2]**2*(w**2 - wlm[2]**2 + 4*wtq**2) - 2j*w*tlm[2] - 1)*np.sin(2*phid)+\
        4*tlm[2]*wtq*(1+1j*w*tlm[2])*np.cos(2*phid)
    Ib3_3 = (tlm[2]*(w-wlm[2]-2*wtq) - 1j)*(tlm[2]*(w + wlm[2] -2*wtq)-1j)*(tlm[2]*(w-wlm[2]+2*wtq)-1j)*(tlm[2]*(w+wlm[2]+2*wtq)-1j)
    Ib_33 = (Ib1_3 * np.cos(philm) - Ib2_3 * tlm[2] * wlm[2] *np.sin(philm))*tlm[2]/Ib3_3
    #这是第四个44模式
    Ib1_4 = 2*tlm[3] *wtq*(tlm[3]**2 *(w**2 + wlm[3]**2 -4*wtq**2) - 2*1j*w*tlm[3] - 1)*np.cos(2*phid) - \
        (1+1j*w*tlm[3])*(tlm[3]**2*(w**2 - wlm[3]**2 - 4*wtq**2) - 2j * w *tlm[3] - 1)*np.sin(2*phid)
    Ib2_4 = (tlm[3]**2*(w**2 - wlm[3]**2 + 4*wtq**2) - 2j*w*tlm[3] - 1)*np.sin(2*phid)+\
        4*tlm[3]*wtq*(1+1j*w*tlm[3])*np.cos(2*phid)
    Ib3_4 = (tlm[3]*(w-wlm[3]-2*wtq) - 1j)*(tlm[3]*(w + wlm[3] -2*wtq)-1j)*(tlm[3]*(w-wlm[3]+2*wtq)-1j)*(tlm[3]*(w+wlm[3]+2*wtq)-1j)
    Ib_44 = (Ib1_4 * np.cos(philm) - Ib2_4 * tlm[3] * wlm[3] *np.sin(philm))*tlm[3]/Ib3_4

    return Ib_22,Ib_21,Ib_33,Ib_44

#这里给出天琴探测器响应后的频域波形
#没准在这里给出一些参数，将Ib_p确定下来

gamma = pi/3
thetad = 1.17662
psid = -0.873302
phid = -0.615727
#Mz = M 
#Dl = r

'''
#这里是频率下的波形
def sf(freq):
    phid1 = phid
    phid2 = phid +pi/4
    freq = freq*2*pi #将圆频率转化到频率
    #这一部分已与原文对比，形式上没有问题
    sig_freq_22 = (-0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.cos(2*psid)*Ib_p(freq,wlm,tlm,phid1)[0]+\
                np.sin(gamma)*np.cos(thetad)*np.sin(2*psid)*Ib_p(freq,wlm,tlm,phid2)[0])*Mz*A1*Yp1/Dl +\
                (0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.sin(2*psid)*Ib_c(freq,wlm,tlm,phid1)[0]+\
                np.sin(gamma)*np.cos(thetad)*np.cos(2*psid)*Ib_c(freq,wlm,tlm,phid2)[0])*Mz*A1*Yc1/Dl
    
    sig_freq_21 = (-0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.cos(2*psid)*Ib_p(freq,wlm,tlm,phid1)[1]+\
                np.sin(gamma)*np.cos(thetad)*np.sin(2*psid)*Ib_p(freq,wlm,tlm,phid2)[1])*Mz*A2*Yp2/Dl +\
                (0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.sin(2*psid)*Ib_c(freq,wlm,tlm,phid1)[1]+\
                np.sin(gamma)*np.cos(thetad)*np.cos(2*psid)*Ib_c(freq,wlm,tlm,phid2)[1])*Mz*A2*Yc2/Dl
    
    sig_freq_33 = (-0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.cos(2*psid)*Ib_p(freq,wlm,tlm,phid1)[2]+\
                np.sin(gamma)*np.cos(thetad)*np.sin(2*psid)*Ib_p(freq,wlm,tlm,phid2)[2])*Mz*A3*Yp3/Dl +\
                (0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.sin(2*psid)*Ib_c(freq,wlm,tlm,phid1)[2]+\
                np.sin(gamma)*np.cos(thetad)*np.cos(2*psid)*Ib_c(freq,wlm,tlm,phid2)[2])*Mz*A3*Yc3/Dl
    
    sig_freq_44 = (-0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.cos(2*psid)*Ib_p(freq,wlm,tlm,phid1)[3]+\
                np.sin(gamma)*np.cos(thetad)*np.sin(2*psid)*Ib_p(freq,wlm,tlm,phid2)[3])*Mz*A4*Yp4/Dl +\
                (0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.sin(2*psid)*Ib_c(freq,wlm,tlm,phid1)[3]+\
                np.sin(gamma)*np.cos(thetad)*np.cos(2*psid)*Ib_c(freq,wlm,tlm,phid2)[3])*Mz*A4*Yc4/Dl
    return sig_freq_22+sig_freq_21+sig_freq_33+sig_freq_44
'''
def fin(Mass,massratio):
    M = Mass * TSUN
    mass_ratio = massratio
    eta = mass_ratio/(1+mass_ratio)**2   #eta为对称质量比
    j = eta*(2*np.sqrt(3)-3.5171*eta+2.5763*eta**2)

    ################################################################
    w1 = (f1[1] + f2[1]*(1-j)**f3[1])/M ; dw1 = 0
    w2 = (f1[2] + f2[2]*(1-j)**f3[2])/M ; dw2 = 0
    w3 = (f1[3] + f2[3]*(3-j)**f3[3])/M ; dw3 = 0
    w4 = (f1[4] + f2[4]*(4-j)**f3[4])/M ; dw4 = 0
    tau1 = 2*(q1[1]+q2[1]*(1-j)**q3[1])/w1 ; dtau1 = 0
    tau2 = 2*(q1[2]+q2[2]*(1-j)**q3[2])/w2 ; dtau2 = 0
    tau3 = 2*(q1[3]+q2[3]*(1-j)**q3[3])/w3 ; dtau3 = 0
    tau4 = 2*(q1[4]+q2[4]*(1-j)**q3[4])/w4 ; dtau4 = 0
    #return max(0.5*w2/(2*pi),2*1e-5)
    
    return 0.5*w2/(2*pi)

def fout(Mass,massratio):
    M = Mass * TSUN
    mass_ratio = massratio
    eta = mass_ratio/(1+mass_ratio)**2   #eta为对称质量比
    j = eta*(2*np.sqrt(3)-3.5171*eta+2.5763*eta**2)

    ################################################################
    w1 = (f1[1] + f2[1]*(1-j)**f3[1])/M ; dw1 = 0
    w2 = (f1[2] + f2[2]*(1-j)**f3[2])/M ; dw2 = 0
    w3 = (f1[3] + f2[3]*(3-j)**f3[3])/M ; dw3 = 0
    w4 = (f1[4] + f2[4]*(4-j)**f3[4])/M ; dw4 = 0
    tau1 = 2*(q1[1]+q2[1]*(1-j)**q3[1])/w1 ; dtau1 = 0
    tau2 = 2*(q1[2]+q2[2]*(1-j)**q3[2])/w2 ; dtau2 = 0
    tau3 = 2*(q1[3]+q2[3]*(1-j)**q3[3])/w3 ; dtau3 = 0
    tau4 = 2*(q1[4]+q2[4]*(1-j)**q3[4])/w4 ; dtau4 = 0
    return 2*(w4/2/pi)

def sf(freq,paramater,delta_w,delta_tau,iot = pi/4):
    M = paramater[0] * TSUN
    Mz = M
    mass_ratio = paramater[1]
    r = DL(paramater[2]) * MPC
    Dl = r


    dw1,dw2,dw3,dw4 = delta_w[0],delta_w[1],delta_w[2],delta_w[3]
    dtau1,dtau2,dtau3,dtau4 = delta_tau[0],delta_tau[1],delta_tau[2],delta_tau[3]


    dM = 0
    t0 = 0
    dr = 0
    phi = 0

    #这里放置波形一般性的所有内容
    m1=2;m2=1;m3=3;m4=4
    #这一块检查过没问题
    #四个模式的频率和衰减时间
    #这里1234，分别代表22,21,33,44模式

    #q = M1/M2 （eta>=1），因此是无量纲数，就自由取值吧

    eta = mass_ratio/(1+mass_ratio)**2   #eta为对称质量比
    j = eta*(2*np.sqrt(3)-3.5171*eta+2.5763*eta**2)

    ################################################################
    w1 = (f1[1] + f2[1]*(1-j)**f3[1])/M *(1+dw1); 
    w2 = (f1[2] + f2[2]*(1-j)**f3[2])/M *(1+dw2); 
    w3 = (f1[3] + f2[3]*(3-j)**f3[3])/M *(1+dw3); 
    w4 = (f1[4] + f2[4]*(4-j)**f3[4])/M *(1+dw4); 
    tau1 = 2*(q1[1]+q2[1]*(1-j)**q3[1])/w1 *(1+dtau1); 
    tau2 = 2*(q1[2]+q2[2]*(1-j)**q3[2])/w2 *(1+dtau2); 
    tau3 = 2*(q1[3]+q2[3]*(1-j)**q3[3])/w3 *(1+dtau3); 
    tau4 = 2*(q1[4]+q2[4]*(1-j)**q3[4])/w4 *(1+dtau4); 
    wlm = np.array([w1,w2,w3,w4])
    tlm = np.array([tau1,tau2,tau3,tau4])

    #四个模式的振幅,现在进行定义非自旋amplitude,参考文章： PHYS.REV.D97,044048(2018)
    A1 = eta*0.571+0.303#22
    A2 = eta*0.099+0.06#21
    A3 = eta*0.157+0.671#33
    A4 = 0.122 -0.188*eta -0.964*eta**2#44

    #球谐函数，检查过没问题
    iot=pi/4
    Yp1 = np.sqrt(5/(4*pi)) * (1+np.cos(iot)**2)/2 #22
    Yp2 = np.sqrt(5/(4*pi)) * np.sin(iot) #21
    Yp3 = -np.sqrt(21/(8*pi)) * (1+np.cos(iot)**2) * np.sin(iot) / 2 #33
    Yp4 = np.sqrt(63/(16*pi))*(1+np.cos(iot)**2) * np.sin(iot)**2/2 #44
    Yc1 = np.sqrt(5/(4*pi)) * np.cos(iot)
    Yc2 = np.sqrt(5/(4*pi)) * np.cos(iot) * np.sin(iot)
    Yc3 = -np.sqrt(21/(8*pi)) * np.cos(iot) * np.sin(iot)
    Yc4 = np.sqrt(63/(16*pi))* np.cos(iot) * np.sin(iot)**2

    phid1 = phid
    phid2 = phid +pi/4
    
    freq = freq*2*pi #将圆频率转化到频率
    #这一部分已与原文对比，形式上没有问题
    sig_freq_22 = (-0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.cos(2*psid)*Ib_p(freq,wlm,tlm,phid1)[0]+\
                np.sin(gamma)*np.cos(thetad)*np.sin(2*psid)*Ib_p(freq,wlm,tlm,phid2)[0])*Mz*A1*Yp1/Dl +\
                (0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.sin(2*psid)*Ib_c(freq,wlm,tlm,phid1)[0]+\
                np.sin(gamma)*np.cos(thetad)*np.cos(2*psid)*Ib_c(freq,wlm,tlm,phid2)[0])*Mz*A1*Yc1/Dl
    
    sig_freq_21 = (-0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.cos(2*psid)*Ib_p(freq,wlm,tlm,phid1)[1]+\
                np.sin(gamma)*np.cos(thetad)*np.sin(2*psid)*Ib_p(freq,wlm,tlm,phid2)[1])*Mz*A2*Yp2/Dl +\
                (0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.sin(2*psid)*Ib_c(freq,wlm,tlm,phid1)[1]+\
                np.sin(gamma)*np.cos(thetad)*np.cos(2*psid)*Ib_c(freq,wlm,tlm,phid2)[1])*Mz*A2*Yc2/Dl
    
    sig_freq_33 = (-0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.cos(2*psid)*Ib_p(freq,wlm,tlm,phid1)[2]+\
                np.sin(gamma)*np.cos(thetad)*np.sin(2*psid)*Ib_p(freq,wlm,tlm,phid2)[2])*Mz*A3*Yp3/Dl +\
                (0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.sin(2*psid)*Ib_c(freq,wlm,tlm,phid1)[2]+\
                np.sin(gamma)*np.cos(thetad)*np.cos(2*psid)*Ib_c(freq,wlm,tlm,phid2)[2])*Mz*A3*Yc3/Dl
    
    sig_freq_44 = (-0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.cos(2*psid)*Ib_p(freq,wlm,tlm,phid1)[3]+\
                np.sin(gamma)*np.cos(thetad)*np.sin(2*psid)*Ib_p(freq,wlm,tlm,phid2)[3])*Mz*A4*Yp4/Dl +\
                (0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.sin(2*psid)*Ib_c(freq,wlm,tlm,phid1)[3]+\
                np.sin(gamma)*np.cos(thetad)*np.cos(2*psid)*Ib_c(freq,wlm,tlm,phid2)[3])*Mz*A4*Yc4/Dl
    

    return sig_freq_22+sig_freq_21+sig_freq_33+sig_freq_44


def sf_decomposition(freq,paramater,delta_w,delta_tau,iot = pi/4):
    M = paramater[0] * TSUN
    Mz = M
    mass_ratio = paramater[1]
    r = DL(paramater[2]) * MPC
    Dl = r


    dw1,dw2,dw3,dw4 = delta_w[0],delta_w[1],delta_w[2],delta_w[3]
    dtau1,dtau2,dtau3,dtau4 = delta_tau[0],delta_tau[1],delta_tau[2],delta_tau[3]


    dM = 0
    t0 = 0
    dr = 0
    phi = 0

    #这里放置波形一般性的所有内容
    m1=2;m2=1;m3=3;m4=4
    #这一块检查过没问题
    #四个模式的频率和衰减时间
    #这里1234，分别代表22,21,33,44模式

    #q = M1/M2 （eta>=1），因此是无量纲数，就自由取值吧

    eta = mass_ratio/(1+mass_ratio)**2   #eta为对称质量比
    j = eta*(2*np.sqrt(3)-3.5171*eta+2.5763*eta**2)

    ################################################################
    w1 = (f1[1] + f2[1]*(1-j)**f3[1])/M *(1+dw1); 
    w2 = (f1[2] + f2[2]*(1-j)**f3[2])/M *(1+dw2); 
    w3 = (f1[3] + f2[3]*(3-j)**f3[3])/M *(1+dw3); 
    w4 = (f1[4] + f2[4]*(4-j)**f3[4])/M *(1+dw4); 
    tau1 = 2*(q1[1]+q2[1]*(1-j)**q3[1])/w1 *(1+dtau1); 
    tau2 = 2*(q1[2]+q2[2]*(1-j)**q3[2])/w2 *(1+dtau2); 
    tau3 = 2*(q1[3]+q2[3]*(1-j)**q3[3])/w3 *(1+dtau3); 
    tau4 = 2*(q1[4]+q2[4]*(1-j)**q3[4])/w4 *(1+dtau4); 
    wlm = np.array([w1,w2,w3,w4])
    tlm = np.array([tau1,tau2,tau3,tau4])

    #四个模式的振幅,现在进行定义非自旋amplitude,参考文章： PHYS.REV.D97,044048(2018)
    A1 = eta*0.571+0.303#22
    A2 = eta*0.099+0.06#21
    A3 = eta*0.157+0.671#33
    A4 = 0.122 -0.188*eta -0.964*eta**2#44

    #球谐函数，检查过没问题
    iot=pi/4
    Yp1 = np.sqrt(5/(4*pi)) * (1+np.cos(iot)**2)/2 #22
    Yp2 = np.sqrt(5/(4*pi)) * np.sin(iot) #21
    Yp3 = -np.sqrt(21/(8*pi)) * (1+np.cos(iot)**2) * np.sin(iot) / 2 #33
    Yp4 = np.sqrt(63/(16*pi))*(1+np.cos(iot)**2) * np.sin(iot)**2/2 #44
    Yc1 = np.sqrt(5/(4*pi)) * np.cos(iot)
    Yc2 = np.sqrt(5/(4*pi)) * np.cos(iot) * np.sin(iot)
    Yc3 = -np.sqrt(21/(8*pi)) * np.cos(iot) * np.sin(iot)
    Yc4 = np.sqrt(63/(16*pi))* np.cos(iot) * np.sin(iot)**2

    phid1 = phid
    phid2 = phid +pi/4
    
    freq = freq*2*pi #将圆频率转化到频率
    #这一部分已与原文对比，形式上没有问题
    sig_freq_22 = (-0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.cos(2*psid)*Ib_p(freq,wlm,tlm,phid1)[0]+\
                np.sin(gamma)*np.cos(thetad)*np.sin(2*psid)*Ib_p(freq,wlm,tlm,phid2)[0])*Mz*A1*Yp1/Dl +\
                (0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.sin(2*psid)*Ib_c(freq,wlm,tlm,phid1)[0]+\
                np.sin(gamma)*np.cos(thetad)*np.cos(2*psid)*Ib_c(freq,wlm,tlm,phid2)[0])*Mz*A1*Yc1/Dl
    
    sig_freq_21 = (-0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.cos(2*psid)*Ib_p(freq,wlm,tlm,phid1)[1]+\
                np.sin(gamma)*np.cos(thetad)*np.sin(2*psid)*Ib_p(freq,wlm,tlm,phid2)[1])*Mz*A2*Yp2/Dl +\
                (0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.sin(2*psid)*Ib_c(freq,wlm,tlm,phid1)[1]+\
                np.sin(gamma)*np.cos(thetad)*np.cos(2*psid)*Ib_c(freq,wlm,tlm,phid2)[1])*Mz*A2*Yc2/Dl
    
    sig_freq_33 = (-0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.cos(2*psid)*Ib_p(freq,wlm,tlm,phid1)[2]+\
                np.sin(gamma)*np.cos(thetad)*np.sin(2*psid)*Ib_p(freq,wlm,tlm,phid2)[2])*Mz*A3*Yp3/Dl +\
                (0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.sin(2*psid)*Ib_c(freq,wlm,tlm,phid1)[2]+\
                np.sin(gamma)*np.cos(thetad)*np.cos(2*psid)*Ib_c(freq,wlm,tlm,phid2)[2])*Mz*A3*Yc3/Dl
    
    sig_freq_44 = (-0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.cos(2*psid)*Ib_p(freq,wlm,tlm,phid1)[3]+\
                np.sin(gamma)*np.cos(thetad)*np.sin(2*psid)*Ib_p(freq,wlm,tlm,phid2)[3])*Mz*A4*Yp4/Dl +\
                (0.5*np.sin(gamma)*(1+np.cos(thetad)**2)*np.sin(2*psid)*Ib_c(freq,wlm,tlm,phid1)[3]+\
                np.sin(gamma)*np.cos(thetad)*np.cos(2*psid)*Ib_c(freq,wlm,tlm,phid2)[3])*Mz*A4*Yc4/Dl
    

    return sig_freq_22,sig_freq_21,sig_freq_33,sig_freq_44
# Mtot = 1e8*3
# M_s = Mtot*TSUN
# M_ratio = 2
# R_shift = 1
# para = [Mtot,M_ratio,R_shift]
para_dw = [0,0,0,0]
para_dtau = [0,0,0,0]
# fi = fin(Mtot,2)
# fe = fout(Mtot,2)
# freqen = np.arange(1e-6,1,1e-6)
# freq_rd = np.arange(fi,fe,1e-6)
# #np.abs(sf(freq_rd))
# sig_test = sf(freq_rd,para,para_dw,para_dtau)

# eta = M_ratio/(1+M_ratio)**2   #eta为对称质量比
# j = eta*(2*np.sqrt(3)-3.5171*eta+2.5763*eta**2)
# w1 = (f1[1] + f2[1]*(1-j)**f3[1])/M_s ; dw1 = 0
# w2 = (f1[2] + f2[2]*(1-j)**f3[2])/M_s ; dw2 = 0
# w3 = (f1[3] + f2[3]*(3-j)**f3[3])/M_s ; dw3 = 0
# w4 = (f1[4] + f2[4]*(4-j)**f3[4])/M_s ; dw4 = 0
# tau1 = 2*(q1[1]+q2[1]*(1-j)**q3[1])/w1 ; dtau1 = 0
# tau2 = 2*(q1[2]+q2[2]*(1-j)**q3[2])/w2 ; dtau2 = 0
# tau3 = 2*(q1[3]+q2[3]*(1-j)**q3[3])/w3 ; dtau3 = 0
# tau4 = 2*(q1[4]+q2[4]*(1-j)**q3[4])/w4 ; dtau4 = 0
# # print(tau1,tau2,tau3,tau4)
# print(eta)
##########################################################################################
#这里，是加gap的函数，并且画图呈现出加了gap后对时域波源的影响，同时在此处也进行了频域到时域的变换#
##########################################################################################

#################################################
#很好，看起来加随机位置窗的工作已经很好完成了哈哈哈哈#
#################################################
#这里
#现在把内置的随机数放在外面,这样就可以提前给定gap的随机位置，然后在后面的gp_fisher函数处，再引用这个randnum
#np.random.seed(123)
#randnum = 0

# randnum = np.random.uniform(0,1,1)
# def win(peri,htime,Tvec):
#     #注意，这里可能需要将gap的位置避免放置在ringdown信号的边缘
#     #np.random.seed(1)
#     #posi_gap = peri * np.random.uniform(0,1,1)
#     posi_gap = peri * randnum
#     #posi_gap = peri * uni_rand
#     #posi_gap = 2000
#     Tgap = peri/10
#     Ts = posi_gap - Tgap
#     Te = posi_gap + Tgap
#     Tr = peri/10
#     gap_st = htime[:]

#     for nn in range(len(Tvec)):
        
#         if nn >=Ts and nn <= Te:
            
#             gap_st[nn] = 0
#         elif nn <Ts and nn >= Ts-Tr:
#             gap_st[nn] = gap_st[nn]*0.5*(1+np.cos(pi*(nn-Ts-Tr)/Tr))
            
#         elif nn >Te and nn <= Te+Tr:
#             gap_st[nn] = gap_st[nn]*0.5*(1+np.cos(pi*(nn-Te-Tr)/Tr))

#     #print("完成加窗过程，窗的长度为：",Tgap)
#     return gap_st



# #randnum = .8
# Mtot = 1e7
# M_ratio = 2
# R_shift = 1
# para = [Mtot,M_ratio,R_shift]
# freq_ifft = np.arange(1e-5,1,1e-5)
# f_sf = sf(freq_ifft,para,para_dw,para_dtau)
# st=Ga.Freq_ifft(f_sf)
# # plt.plot(st)
# plt.xlim(0,25000)
# #plt.xscale('log')
# plt.show()

# M_s = Mtot*TSUN
# eta = M_ratio/(1+M_ratio)**2   #eta为对称质量比
# j = eta*(2*np.sqrt(3)-3.5171*eta+2.5763*eta**2)
# w1 = (f1[1] + f2[1]*(1-j)**f3[1])/M_s ; dw1 = 0
# w2 = (f1[2] + f2[2]*(1-j)**f3[2])/M_s ; dw2 = 0
# w3 = (f1[3] + f2[3]*(3-j)**f3[3])/M_s ; dw3 = 0
# w4 = (f1[4] + f2[4]*(4-j)**f3[4])/M_s ; dw4 = 0
# tau1 = 2*(q1[1]+q2[1]*(1-j)**q3[1])/w1 ; dtau1 = 0
# tau2 = 2*(q1[2]+q2[2]*(1-j)**q3[2])/w2 ; dtau2 = 0
# tau3 = 2*(q1[3]+q2[3]*(1-j)**q3[3])/w3 ; dtau3 = 0
# tau4 = 2*(q1[4]+q2[4]*(1-j)**q3[4])/w4 ; dtau4 = 0
# print(tau1,tau2,tau3,tau4)

# period = tau4*2*pi
# time = np.arange(0,len(st),1)

# gaped_st = win(period,st,time)
# print(len(st))
# print(len(time))
# print(len(gaped_st))

# plt.plot(time,gaped_st)
# plt.xlim(0,25000)
# plt.show()