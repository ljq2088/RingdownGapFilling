#这里是根据张春雨文章所构建的波形，以及Robson文章的灵敏度曲线，


import numpy as np
import matplotlib.pyplot as plt
import dataprocess as dp

#import waveform_zxy as wf
#import fisher as fh

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


#频域探测器张量
def Dup(ThetaD,PsiD,PhiD,Gamma = pi/3):
    value = ((1+np.cos(ThetaD)**2)*np.cos(2*PhiD+Gamma)-np.sin(ThetaD)**2)*np.cos(2*PsiD)/4 + np.cos(ThetaD)*np.sin(2*PhiD+Gamma)*np.sin(2*PsiD)/2
    return value

def Dvp(ThetaD,PsiD,PhiD,Gamma = pi/3):
    value = ((1+np.cos(ThetaD)**2)*np.cos(2*PhiD - Gamma)-np.sin(ThetaD)**2)*np.cos(2*PsiD)/4 + np.cos(ThetaD)*np.sin(2*PhiD-Gamma)*np.sin(2*PsiD)/2
    return value

def Duc(ThetaD,PsiD,PhiD,Gamma = pi/3):
    value = (np.sin(ThetaD)**2 - (1+np.cos(ThetaD)**2)*np.cos(2*PhiD+Gamma))*np.sin(2*PsiD)/4 + np.cos(ThetaD)*np.sin(2*PhiD+Gamma)*np.cos(2*PsiD)/2
    return value

def Dvc(ThetaD,PsiD,PhiD,Gamma = pi/3):
    value = (np.sin(ThetaD)**2 - (1+np.cos(ThetaD)**2)*np.cos(2*PhiD-Gamma))*np.sin(2*PsiD)/4 + np.cos(ThetaD)*np.sin(2*PhiD-Gamma)*np.cos(2*PsiD)/2
    return value

#转移函数
def transU(f_,ThetaD,PsiD,PhiD,detector=0,Gamma = pi/3):
    f0 = 0.019085380764271725
    if detector == 'tj':
        f0 = 0.015904483970226438
    elif detector == 'tq':
        f0 = 0.2754737430459696

    u = np.array([np.cos(Gamma/2),-np.sin(Gamma/2),0])
    o = np.array([-np.sin(ThetaD)*np.cos(PhiD),-np.sin(ThetaD)*np.sin(PhiD),-np.cos(ThetaD)])
    udo = np.dot(u,o)

    tranferfunction = 0.5*(np.sinc(f_*(1-udo)/2/f0)*np.exp(f_*(3+udo)/(2*1j*f0))+np.sinc(f_*(1+udo)/2/f0)*np.exp(f_*(1+udo)/(2*1j*f0)))
    return tranferfunction


def transV(f_,ThetaD,PsiD,PhiD,detector=0,Gamma = pi/3):
    f0 = 0.019085380764271725
    if detector == 'tj':
        f0 = 0.015904483970226438
    elif detector == 'tq':
        f0 = 0.2754737430459696

    v = np.array([np.cos(Gamma/2),np.sin(Gamma/2),0])
    o = np.array([-np.sin(ThetaD)*np.cos(PhiD),-np.sin(ThetaD)*np.sin(PhiD),-np.cos(ThetaD)])
    vdo = np.dot(v,o)

    tranferfunction = 0.5*(np.sinc(f_*(1-vdo)/2/f0)*np.exp(f_*(3+vdo)/(2*1j*f0))+np.sinc(f_*(1+vdo)/2/f0)*np.exp(f_*(1+vdo)/(2*1j*f0)))
    return tranferfunction

def Ia(f_,wlm,tlm,Philm):
    w = 2*pi*f_
    Ia1 = (1+1j*w*tlm)*(1+2j*w*tlm-tlm**2*(w**2-wlm**2))
    Ia2 = tlm**2*(w**2-wlm**2)-1-2j*w*tlm
    Ia3 = (tlm*(w-wlm)-1j)**2*(tlm*(w+wlm)-1j)**2
    return (Ia1*np.cos(Philm) - Ia2*tlm*wlm*np.sin(Philm))*tlm/Ia3

def hpc(f_,paramater,dw,dtau,Philm = pi/3):
    Mz = paramater[0] * TSUN
    mass_ratio = paramater[1]
    r = DL(paramater[2]) * MPC
    dL = r
    dw1,dw2,dw3,dw4 = dw[0],dw[1],dw[2],dw[3]
    dtau1,dtau2,dtau3,dtau4 = dtau[0],dtau[1],dtau[2],dtau[3]
    eta = mass_ratio/(1+mass_ratio)**2
    j = eta*(2*np.sqrt(3)-3.5171*eta+2.5763*eta**2)

    #这里1234，分别代表22,21,33,44模式
    A1 = eta*0.571+0.303#22
    A2 = eta*0.099+0.06#21
    A3 = eta*0.157+0.671#33
    A4 = 0.122 -0.188*eta -0.964*eta**2#44

    w1 = (f1[1] + f2[1]*(1-j)**f3[1])/Mz *(1+dw1)
    w2 = (f1[2] + f2[2]*(1-j)**f3[2])/Mz *(1+dw2)
    w3 = (f1[3] + f2[3]*(3-j)**f3[3])/Mz *(1+dw3)
    w4 = (f1[4] + f2[4]*(4-j)**f3[4])/Mz *(1+dw4)
    tau1 = 2*(q1[1]+q2[1]*(1-j)**q3[1])/w1 *(1+dtau1)
    tau2 = 2*(q1[2]+q2[2]*(1-j)**q3[2])/w2 *(1+dtau2)
    tau3 = 2*(q1[3]+q2[3]*(1-j)**q3[3])/w3 *(1+dtau3)
    tau4 = 2*(q1[4]+q2[4]*(1-j)**q3[4])/w4 *(1+dtau4)

    iot=pi/4            #iot可以再换个参数，先默认此参数
    Yp1 = np.sqrt(5/(4*pi)) * (1+np.cos(iot)**2)/2 #22
    Yp2 = np.sqrt(5/(4*pi)) * np.sin(iot) #21
    Yp3 = -np.sqrt(21/(8*pi)) * (1+np.cos(iot)**2) * np.sin(iot) / 2 #33
    Yp4 = np.sqrt(63/(16*pi))*(1+np.cos(iot)**2) * np.sin(iot)**2/2 #44

    Philm1 = Philm
    hp1 = A1*Yp1*Ia(f_,w1,tau1,Philm1)
    hp2 = A2*Yp2*Ia(f_,w2,tau2,Philm1)
    hp3 = A3*Yp3*Ia(f_,w3,tau3,Philm1)
    hp4 = A4*Yp4*Ia(f_,w4,tau4,Philm1)

    hp = Mz/dL * (hp1+hp2+hp3+hp4)

    Yc1 = np.sqrt(5/(4*pi)) * np.cos(iot)
    Yc2 = np.sqrt(5/(4*pi)) * np.cos(iot) * np.sin(iot)
    Yc3 = -np.sqrt(21/(8*pi)) * np.cos(iot) * np.sin(iot)
    Yc4 = np.sqrt(63/(16*pi))* np.cos(iot) * np.sin(iot)**2
    
    Philm2 = Philm - pi/2
    hc1 = A1*Yc1*Ia(f_,w1,tau1,Philm2)
    hc2 = A2*Yc2*Ia(f_,w2,tau2,Philm2)
    hc3 = A3*Yc3*Ia(f_,w3,tau3,Philm2)
    hc4 = A4*Yc4*Ia(f_,w4,tau4,Philm2)

    hc = Mz/dL * (hc1+hc2+hc3+hc4)
    return hp,hc

#detec这个参数，默认为lisa探测器的信号，输入字符串'tj'则得到太极的信号，输入字符串'tq'则得到天琴的信号
def sfa(f_,paramater,angles,detec = 0,dw=[0,0,0,0],dtau = [0,0,0,0]):
    ThetaD = angles[0]
    PsiD   = angles[1]
    PhiD   = angles[2]
    sigp   = (Dup(ThetaD,PsiD,PhiD,Gamma = pi/3)*transU(f_,ThetaD,PsiD,PhiD,detec,Gamma = pi/3) - \
            Dvp(ThetaD,PsiD,PhiD,Gamma = pi/3)*transV(f_,ThetaD,PsiD,PhiD,detec,Gamma = pi/3))*hpc(f_,paramater,dw,dtau,Philm = pi/3)[0]
    
    sigc   = (Duc(ThetaD,PsiD,PhiD,Gamma = pi/3)*transU(f_,ThetaD,PsiD,PhiD,detec,Gamma = pi/3) - \
            Dvc(ThetaD,PsiD,PhiD,Gamma = pi/3)*transV(f_,ThetaD,PsiD,PhiD,detec,Gamma = pi/3))*hpc(f_,paramater,dw,dtau,Philm = pi/3)[1]
    
    return sigp+sigc

#这两个是ringdown开始和结束的频率
def fin(Mass,massratio):
    M = Mass * TSUN
    mass_ratio = massratio
    eta = mass_ratio/(1+mass_ratio)**2   #eta为对称质量比
    j = eta*(2*np.sqrt(3)-3.5171*eta+2.5763*eta**2)
    w2 = (f1[2] + f2[2]*(1-j)**f3[2])/M ; dw2 = 0
    return 0.5*w2/(2*pi)

def fout(Mass,massratio):
    M = Mass * TSUN
    mass_ratio = massratio
    eta = mass_ratio/(1+mass_ratio)**2   #eta为对称质量比
    j = eta*(2*np.sqrt(3)-3.5171*eta+2.5763*eta**2)
    w4 = (f1[4] + f2[4]*(4-j)**f3[4])/M ; dw4 = 0
    return 2*(w4/2/pi)


def PSD_ls(f):
    
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """

    L = 2.5*10**9   # Length of LISA arm
    f0 = 0.019085380764271725   
    
    Poms = ((1.5*10**-11)**2)*(1 + ((2*10**-3)/f)**4)  # Optical Metrology Sensor
    Pacc = (3*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
                                            + np.tanh(1680*(0.00215 - f)))   # Confusion noise
    alpha = 0.171
    beta = 292
    k =1020
    gamma = 1680
    f_k = 0.00215 
    PSD = ((10/(3*L*L))*(Poms + (4*Pacc)/(np.power(2*np.pi*f,4)))*(1 + 0.6*(f/f0)*(f/f0)) + Sc) # PSD
        
    # Handling the zeroth frequency bin
    
    where_are_NaNs = np.isnan(PSD) 
    PSD[where_are_NaNs] = 1e100    # at f = 0, PSD is infinite. Approximate infinity and burn my 
                                   # mathematics degree to the ground. 
    
    return PSD

def PSD_tj(f):
    
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """

    L = 3.0*10**9   # Length of Taiji arm
    f0 = 0.015904483970226438    
    
    Poms = ((8*10**-12)**2)*(1 + ((2*10**-3)/f)**4)  # Optical Metrology Sensor
    Pacc = (3*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
                                            + np.tanh(1680*(0.00215 - f)))   # Confusion noise
    alpha = 0.171
    beta = 292
    k =1020
    gamma = 1680
    f_k = 0.00215 
    PSD = ((10/(3*L*L))*(Poms + (4*Pacc)/(np.power(2*np.pi*f,4)))*(1 + 0.6*(f/f0)*(f/f0)) +Sc) # PSD
        
    # Handling the zeroth frequency bin
    
    where_are_NaNs = np.isnan(PSD) 
    PSD[where_are_NaNs] = 1e100    # at f = 0, PSD is infinite. Approximate infinity and burn my 
                                   # mathematics degree to the ground. 
    
    return PSD

def PSD_tq(f):
    #天琴不考虑前景噪声
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """

    L = np.sqrt(3)*10**8   # Length of LISA arm
    f0 = 0.2754737430459696   
    
    Poms = ((1.0*10**-12)**2)*(1 + ((2*10**-3)/f)**4)  # Optical Metrology Sensor
    Pacc = (1*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
                                            + np.tanh(1680*(0.00215 - f)))   # Confusion noise
    alpha = 0.171
    beta = 292
    k =1020
    gamma = 1680
    f_k = 0.00215 
    PSD = ((10/(3*L*L))*(Poms + (4*Pacc)/(np.power(2*np.pi*f,4)))*(1 + 0.6*(f/f0)*(f/f0)) ) # PSD
        
    # Handling the zeroth frequency bin
    
    where_are_NaNs = np.isnan(PSD) 
    PSD[where_are_NaNs] = 1e100    # at f = 0, PSD is infinite. Approximate infinity and burn my 
                                   # mathematics degree to the ground. 
    
    return PSD


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

#这个是第二款波形
def sfb(freq,paramater,delta_w,delta_tau,iot = pi/4):
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


#########################################################################################################################################
################################################这一部分是构建加了数据间隙的波形############################################################
#########################################################################################################################################
def gap_sfa(f_,paramater,angles,random = 0.5,detec = 0,dw=[0,0,0,0],dtau = [0,0,0,0]):
    gsfa_mass = paramater[0]
    gsfa_massratio = paramater[1]
    origin_freq_sig = sfa(f_,paramater,angles,detec,dw,dtau)
    origin_time_sig = dp.Freq_ifft(origin_freq_sig)
    win_time_sig = dp.win(2*pi*dp.qnm_t4(paramater),origin_time_sig,random)
    win_freq_sig = dp.Time_fft(win_time_sig)
    nn = 0
    while f_[nn]<fin(gsfa_mass,gsfa_massratio):
        nn+=1
    n_s = nn
    while f_[nn]<fout(gsfa_mass,gsfa_massratio):
        nn+=1
    n_e = nn
    freq_rd_gap = f_[n_s:n_e]
    win_freq_sig_rd = win_freq_sig[n_s:n_e]
    origin_freq_sig_rd = origin_freq_sig[n_s:n_e]
    return freq_rd_gap,win_freq_sig_rd,origin_freq_sig_rd

def gap_sfa_fm(f_,paramater,angles,n_ss,n_ee,random = 0.5,detec = 0,dw=[0,0,0,0],dtau = [0,0,0,0]):
    #这个是为了计算fisher矩阵的
    origin_freq_sig = sfa(f_,paramater,angles,detec,dw,dtau)
    origin_time_sig = dp.Freq_ifft(origin_freq_sig)
    win_time_sig = dp.win(2*pi*dp.qnm_t4(paramater),origin_time_sig,random)
    win_freq_sig = dp.Time_fft(win_time_sig)

    win_freq_sig_rd = win_freq_sig[n_ss:n_ee]
    
    return np.array(win_freq_sig_rd)