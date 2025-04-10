import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter,stft
import waveform_spin as wf
plt.rcParams['agg.path.chunksize'] = 10000


# Bunch of units

GM_sun = 1.3271244*1e20
c =2.9979246*1e8
M_sun =1.9884099*1e30
G = 6.6743*1e-11
pc= 3.0856776*1e16
pi = np.pi
Mpc = (10**6) * pc



#这里直接给以秒为单位的情况
C       = 299792458.         # m/s
YEAR    = 3.15581497632e7    # sec
TSUN    = 4.92549232189886339689643862e-6 # mass of sun in seconds (G=C=1)
MPC     = 3.08568025e22/C    # mega-Parsec in seconds

def inner_prod(sig1_f,sig2_f,PSD,delta_f):
    """
    Inputs:
    sig1_f, sig2_f are continuous time fourier transforms with dimensions of seconds.
    PSD (power spectral density) defined in the function below. 
    delta_f : spacing of fourier frequencies
    
    outputs: Standard inner product, dimensionless.
    """

    return (4*delta_f) * np.real(sum(sig1_f*np.conjugate(sig2_f)/PSD))

def SNR(sig1_f,sig2_f,PSD,delta_f):
    """
    Inputs:
    sig1_f, sig2_f are continuous time fourier transforms with dimensions of seconds.
    PSD (power spectral density) defined in the function below. 
    delta_f : spacing of fourier frequencies
    
    outputs: Standard inner product, dimensionless.
    """
    Sigtonoise_ratio = np.sqrt((4*delta_f)  * np.real(sum(sig1_f*np.conjugate(sig2_f)/PSD)))
    return Sigtonoise_ratio
#######################################################################################################################
#这里是将频域的信号，通过一个小trick逆傅里叶变换到纯实的时域波形
def Freq_ifft(h_f_1):
    Htilde = h_f_1
    n = len(h_f_1)
    N = n*2
    #构造用于ifft的频谱信号
    Htilde1 = [0]
    Htilde2 = Htilde[0:-1]*N/2
    Htilde3 = Htilde[-1]*N
    Htilde4 = np.conjugate(Htilde[0:-1])*N/2
    #将Htilde4数组反转，以使之对于中间点对称
    Htilde4 = Htilde4[::-1]

    #依次赋值每个数据点
    Htilde_pre = np.zeros(N,dtype=complex)
    for I in range(n-1):
        Htilde_pre[I+1] = Htilde2[I]
    Htilde_pre[n] = Htilde3
    for I in range(n-1):
        Htilde_pre[n+I+1] = Htilde4[I]

    #逆傅里叶变换
    htime = np.fft.ifft(Htilde_pre)

    return htime


#重新变换到频域,并画出频域的图

def Time_fft(win_htime):
    n = len(win_htime)//2
    Htilde_win_0 = np.fft.fft(win_htime)/n
    Htilde_win = []
    #Htilde_win_0 = Htilde_win_0.tolist
    for I in range(n):
        value = Htilde_win_0[I]
        Htilde_win.append (value)
    #Htilde_win[0] = h_f_1[0]
    #Htilde_win[-1] = h_f_1[-1]

    #**************这里注释的位置是将所有大于初始频谱的gap后的频谱赋值为初始频谱的大小**************#
    #for n in range(n):
    #    if np.abs(Htilde_win[n]) > np.abs(h_f_1[n]):
    #        fuzhi1 = h_f_1[n]
    #        Htilde_win[n] = fuzhi1
    #*******************************************************************************************#
    return Htilde_win
#######################################################################################################################

'''
def Time_fft(h_f_1,win_htime):
    n = len(h_f_1)
    print('hf1:',n)
    print('ht1:',len(win_htime))
    Htilde_win_0 = np.fft.fft(win_htime)/n
    Htilde_win = []
    #Htilde_win_0 = Htilde_win_0.tolist
    for I in range(n):
        value = Htilde_win_0[I]
        Htilde_win.append (value)
    Htilde_win[0] = h_f_1[0]
    Htilde_win[-1] = h_f_1[-1]
    return Htilde_win
'''

#######################################################################################################################
#######################################加入数据间隙#####################################################################
#######################################################################################################################

def qnm_t4(paramater):
    Mz = 10**(paramater[0]) * TSUN
    eta = paramater[1]


    f1 = [0,1.5251 , 0.6000 , 1.8956 , 2.3000]
    f2 = [0,-1.1568 , -0.2339 , -1.3043 , -1.5056]
    f3 = [0,0.1292, 0.4175, 0.1818, 0.2244]
    q1 = [0,0.7, -0.3, 0.9, 1.1929]
    q2 = [0,1.4187, 2.3561, 2.3430, 3.1191]
    q3 = [0,-0.4990, -0.2277, -0.4810, -0.4825]


    j = eta*(2*np.sqrt(3)-3.5171*eta+2.5763*eta**2)

    #w1 = (f1[1] + f2[1]*(1-j)**f3[1])/Mz 
    #w2 = (f1[2] + f2[2]*(1-j)**f3[2])/Mz 
    #w3 = (f1[3] + f2[3]*(1-j)**f3[3])/Mz 
    w4 = (f1[4] + f2[4]*(4-j)**f3[4])/Mz 
    #tau1 = 2*(q1[1]+q2[1]*(1-j)**q3[1])/w1 
    #tau2 = 2*(q1[2]+q2[2]*(1-j)**q3[2])/w2 
    #tau3 = 2*(q1[3]+q2[3]*(1-j)**q3[3])/w3
    tau4 = 2*(q1[4]+q2[4]*(1-j)**q3[4])/w4 
    return tau4
def win(peri,htime,randnum=0.5):
    #默认加窗位置为时域波形正中间
    posi_gap = peri * randnum
    Tgap = peri/10
    Ts = posi_gap - Tgap
    Te = posi_gap + Tgap
    Tr = peri/10
    gap_st = htime[:]
    #这里的lence之所以能这样设置，只是因为ringdown的特性，而不是所有的都可以
    for nn in range(len(htime)):
        
        if nn >=Ts and nn <= Te:
            
            gap_st[nn] = 0
        elif nn <Ts and nn >= Ts-Tr:
            gap_st[nn] = gap_st[nn]*0.5*(1+np.cos(pi*(nn-Ts-Tr)/Tr))
            
        elif nn >Te and nn <= Te+Tr:
            gap_st[nn] = gap_st[nn]*0.5*(1+np.cos(pi*(nn-Te-Tr)/Tr))

    #print("完成加窗过程，窗的长度为：",Tgap)
    return gap_st