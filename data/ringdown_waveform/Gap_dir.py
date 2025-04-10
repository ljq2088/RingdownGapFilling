import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter,stft
plt.rcParams['agg.path.chunksize'] = 10000


# Bunch of units

GM_sun = 1.3271244*1e20
c =2.9979246*1e8
M_sun =1.9884099*1e30
G = 6.6743*1e-11
pc= 3.0856776*1e16
pi = np.pi
Mpc = (10**6) * pc


def SNR_cal(sig1_f,sig2_f,PSD,delta_f):
    """
    Inputs:
    sig1_f, sig2_f are continuous time fourier transforms with dimensions of seconds.
    PSD (power spectral density) defined in the function below. 
    delta_f : spacing of fourier frequencies
    
    outputs: Standard inner product, dimensionless.
    """
    Sigtonoise_ratio = np.sqrt((4*delta_f)  * np.real(sum(sig1_f*np.conjugate(sig2_f)/PSD)))
    print("SNR的大小为:",Sigtonoise_ratio)
    return Sigtonoise_ratio

def PowerSpectralDensity(f):
    
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """

    L = 2.5*10**9   # Length of LISA arm
    f0 = 19.09*10**-3    
    
    Poms = ((1.5*10**-11)**2)*(1 + ((2*10**-3)/f)**4)  # Optical Metrology Sensor
    Pacc = (3*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise
    Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
                                            + np.tanh(1680*(0.00215 - f)))   # Confusion noise
    alpha = 0.171
    beta = 292
    k =1020
    gamma = 1680
    f_k = 0.00215 
    PSD = ((10/(3*L*L))*(Poms + (4*Pacc)/(np.power(2*np.pi*f,4)))*(1 + 0.6*(f/f0)*(f/f0)) ) + Sc # PSD
        
    # Handling the zeroth frequency bin
    
    where_are_NaNs = np.isnan(PSD) 
    PSD[where_are_NaNs] = 1e100    # at f = 0, PSD is infinite. Approximate infinity and burn my 
                                   # mathematics degree to the ground. 
    
    return PSD

def PSD_plot(freqency,PSD,signal):
    plt.plot(freqency,signal)
    plt.plot(freqency,PSD)
    plt.xscale('log');plt.yscale('log')
    plt.show()
    return 0


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

    #plt.plot(np.real(htime))
    #plt.plot(np.imag(htime))
    #plt.xscale('log')
   
    #plt.show()

    #plt.plot(np.abs(Htilde_pre))
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.show()
    return htime



'''
窗函数，现在使用的文献是：PHYSICAL REVIEW D 104, 044035 (2021)
现在是自己diy一个75%的占空比，用数组个数为基准
'''
lengap = 12600*1
#value1 = 567000 + lengap*3*3
value1 = 604800 - lengap*3
Ts = value1 + lengap         #gap start time
Ttr = lengap        #gap transfer time
Te = lengap + Ts         #gap end time
oneweek = 604800

#先写一个周期
def Window(t):

    if t < value1:
        return 1
    elif t < Ts:
        fundown = 0.5*(1+np.cos(pi*(t-Ts-Ttr)/Ttr))
        return fundown
    elif t < Te:
        return 0
    elif t <= oneweek:
        funup = 0.5*(1+np.cos(pi*(t-Te-Ttr)/Ttr))
        return funup
#窗函数具体作用过程
def Wined(SIGNAL,correstime):
    outputsignal = SIGNAL[:]
    for n in range(0,len(SIGNAL)-1,1):
        if correstime[n] <  oneweek:
            cycletime = correstime[n] 
            outputsignal[n] = Window(cycletime)*outputsignal[n]
        else:
            cycletime = correstime[n]  % oneweek
            outputsignal[n] = Window(cycletime) * outputsignal[n]
    
    return outputsignal



#重新变换到频域,并画出频域的图
def Time_fft(h_f_1,win_htime):
    n = len(h_f_1)
    Htilde_win_0 = np.fft.fft(win_htime)/n
    Htilde_win = []
    #Htilde_win_0 = Htilde_win_0.tolist
    for I in range(n):
        value = Htilde_win_0[I]
        Htilde_win.append (value)
    Htilde_win[0] = h_f_1[0]
    Htilde_win[-1] = h_f_1[-1]

    #**************这里注释的位置是将所有大于初始频谱的gap后的频谱赋值为初始频谱的大小**************#
    #for n in range(n):
    #    if np.abs(Htilde_win[n]) > np.abs(h_f_1[n]):
    #        fuzhi1 = h_f_1[n]
    #        Htilde_win[n] = fuzhi1
    #*******************************************************************************************#
    return Htilde_win

def gain_Gaped_Fdata(Hp,Time_final):
    Ht = Freq_ifft(Hp)
    t0 = 1.0
    N = len(Hp)*2
    time_vec = np.linspace(t0,Time_final,N)
    wd_Ht = Wined(Ht,time_vec)

    Hp_gap = Time_fft(Hp,wd_Ht)

    return Hp_gap
