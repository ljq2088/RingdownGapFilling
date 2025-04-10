#这里先浅放一个计算联合观测fisher矩阵的代码
import numpy as np
from waveform_zxy import sf,fin,fout

def inner_prod(sig1_f,sig2_f,PSD,delta_f):
    """
    Inputs:
    sig1_f, sig2_f are continuous time fourier transforms with dimensions of seconds.
    PSD (power spectral density) defined in the function below. 
    delta_f : spacing of fourier frequencies
    
    outputs: Standard inner product, dimensionless.
    """
    return (4*delta_f)  * np.real(sum(sig1_f*np.conjugate(sig2_f)/PSD))

def psd_ls(f):
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """
    #这里现在使用天琴的参数
    L = 2.5*10**9   # Length of LISA arm
    f0 = 0.01908538063694777
    
    Poms = ((1.5*10**-11)**2)*(1 + ((2*10**-3)/f)**4)  # Optical Metrology Sensor
    Pacc = (3.*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise
    #Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
    #                                        + np.tanh(1680*(0.00215 - f)))   # Confusion noise
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

def psd_tq(f):
    
    """
    From https://arxiv.org/pdf/1803.01944.pdf. 
    """
    #这里现在使用天琴的参数
    L = np.sqrt(3)*10**8   # Length of LISA arm
    f0 = 0.27547374120820667
    
    Poms = ((1.*10**-12)**2)*(1 + ((2*10**-3)/f)**4)  # Optical Metrology Sensor
    Pacc = (1.*10**-15)**2*(1 + (4*10**-3/(10*f))**2)*(1 + (f/(8*10**-3))**4)  # Acceleration Noise
    #Sc = 9*10**(-45)*f**(-7/3)*np.exp(-f**0.171 + 292*f*np.sin(1020*f)) * (1 \
    #                                        + np.tanh(1680*(0.00215 - f)))   # Confusion noise
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



def cb_fisher(Mtotw,M_ratiow,R_shiftw):

   delta_w = [0,0,0,0]
   delta_tau = [0,0,0,0]

   f_ini = fin(Mtotw,M_ratiow)
   f_end = fout(Mtotw,M_ratiow)
   freq = np.arange(f_ini,f_end,1e-5)

   para_dw = [0,0,0,0]
   para_dtau = [0,0,0,0]
   para_dtau = delta_tau
   params_1_p = [Mtotw,M_ratiow,R_shiftw]
   params_1_m = [Mtotw,M_ratiow,R_shiftw]
   mode_delta = 1e-3   #懒得改了，这里所有的delta都用22表示把
   omega = delta_w[:]
   omega[0] = omega[0]+mode_delta
   delta_w22_p = omega
   omega = delta_w[:]
   omega[0] = omega[0]-mode_delta
   delta_w22_m = omega
   deriv_w22 = (sf(freq,params_1_p,delta_w22_p,para_dtau) - sf(freq,params_1_m,delta_w22_m,para_dtau))/(2* mode_delta)


   mode_delta = 1e-3   #懒得改了，这里所有的delta都用22表示把
   omega = delta_w[:]
   omega[1] = omega[1]+mode_delta
   delta_w22_p = omega
   omega = delta_w[:]
   omega[1] = omega[1]-mode_delta
   delta_w22_m = omega
   deriv_w21 = (sf(freq,params_1_p,delta_w22_p,para_dtau) - sf(freq,params_1_m,delta_w22_m,para_dtau))/(2* mode_delta)


   mode_delta = 1e-3   #懒得改了，这里所有的delta都用22表示把
   omega = delta_w[:]
   omega[2] = omega[2]+mode_delta
   delta_w22_p = omega
   omega = delta_w[:]
   omega[2] = omega[2]-mode_delta
   delta_w22_m = omega
   deriv_w33 = (sf(freq,params_1_p,delta_w22_p,para_dtau) - sf(freq,params_1_m,delta_w22_m,para_dtau))/(2* mode_delta)

   mode_delta = 1e-3   #懒得改了，这里所有的delta都用22表示把
   omega = delta_w[:]
   omega[3] = omega[3]+mode_delta
   delta_w22_p = omega
   omega = delta_w[:]
   omega[3] = omega[3]-mode_delta
   delta_w22_m = omega
   deriv_w44 = (sf(freq,params_1_p,delta_w22_p,para_dtau) - sf(freq,params_1_m,delta_w22_m,para_dtau))/(2* mode_delta)

   #这里按顺序是tau的各模式
       #22模式的w和tau:



   mode_delta = 1e-3
   tau = delta_tau[:]
   tau[0] = tau[0]+mode_delta
   delta_tau22_p = tau
   tau = delta_tau[:]
   tau[0] = tau[0]-mode_delta
   delta_tau22_m = tau
   deriv_tau22 = (sf(freq,params_1_p,para_dw,delta_tau22_p) - sf(freq,params_1_m,para_dw,delta_tau22_m))/(2* mode_delta)


   #21模式的w和tau:




   mode_delta = 1e-3
   tau = delta_tau[:]
   tau[1] = tau[1]+mode_delta
   delta_tau22_p = tau
   tau = delta_tau[:]
   tau[1] = tau[1]-mode_delta
   delta_tau22_m = tau
   deriv_tau21 = (sf(freq,params_1_p,para_dw,delta_tau22_p) - sf(freq,params_1_m,para_dw,delta_tau22_m))/(2* mode_delta)

   #33模式的w和tau:



   mode_delta = 1e-3
   tau = delta_tau[:]
   tau[2] = tau[2]+mode_delta
   delta_tau22_p = tau
   tau = delta_tau[:]
   tau[2] = tau[2]-mode_delta
   delta_tau22_m = tau
   deriv_tau33 = (sf(freq,params_1_p,para_dw,delta_tau22_p) - sf(freq,params_1_m,para_dw,delta_tau22_m))/(2* mode_delta)

   #44模式的w和tau:



   mode_delta = 1e-3
   tau = delta_tau[:]
   tau[3] = tau[3]+mode_delta
   delta_tau22_p = tau
   tau = delta_tau[:]
   tau[3] = tau[3]-mode_delta
   delta_tau22_m = tau
   deriv_tau44 = (sf(freq,params_1_p,para_dw,delta_tau22_p) - sf(freq,params_1_m,para_dw,delta_tau22_m))/(2* mode_delta)







   diff_vec = [deriv_w22,deriv_w21,deriv_w33,deriv_w44,deriv_tau22,deriv_tau21,deriv_tau33,deriv_tau44]  # Concatenate derivatives

   N_sig = 1  # Number of signals

   N_params = len(diff_vec)  # Number of parameters we care about

   K = N_sig * N_params  # Dimension of Fisher Matrix

   fish_mix_tq = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         fish_mix_tq[i,j] = inner_prod(diff_vec[i],diff_vec[j],psd_tq(freq),1e-5)  # Construct Fisher Matrix
   fish_mix_ls = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         fish_mix_ls[i,j] = inner_prod(diff_vec[i],diff_vec[j],psd_ls(freq),1e-5)          
     

   import mpmath as mp  # Import arbitrary precision matrix
   #mp.dps表示精确度达到的位数
   mp.dps = 4000;   
   #print(fish_mix)
   fish_mix_prec = mp.matrix(fish_mix_tq)+mp.matrix(fish_mix_ls)
   #print(fish_mix_prec)
   fish_mix_inv = fish_mix_prec**-1
   #print(fish_mix_inv)

   #将这三行向量合成为一个矩阵
   Cov_Matrix = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         Cov_Matrix[i,j] = float(fish_mix_inv[i,j])
   print(Cov_Matrix)
   print('Delta logMchirp optimal:',np.sqrt(np.diag(Cov_Matrix))[0])
   print('Delta eta optimal:',np.sqrt(np.diag(Cov_Matrix))[1])
   print('Delta beta optimal:',np.sqrt(np.diag(Cov_Matrix))[2])
   print('Delta beta optimal:',np.sqrt(np.diag(Cov_Matrix))[3])
   return np.sqrt(np.diag(Cov_Matrix))[0],np.sqrt(np.diag(Cov_Matrix))[1],np.sqrt(np.diag(Cov_Matrix))[2],np.sqrt(np.diag(Cov_Matrix))[3],\
   np.sqrt(np.diag(Cov_Matrix))[4],np.sqrt(np.diag(Cov_Matrix))[5],np.sqrt(np.diag(Cov_Matrix))[6],np.sqrt(np.diag(Cov_Matrix))[7]



def cbgp_fisher(Mtotw,M_ratiow,R_shiftw):

   delta_w = [0,0,0,0]
   delta_tau = [0,0,0,0]

   f_ini = fin(Mtotw,M_ratiow)
   f_end = fout(Mtotw,M_ratiow)
   freq = np.arange(f_ini,f_end,1e-5)

   para_dw = [0,0,0,0]
   para_dtau = [0,0,0,0]
   para_dtau = delta_tau
   params_1_p = [Mtotw,M_ratiow,R_shiftw]
   params_1_m = [Mtotw,M_ratiow,R_shiftw]
   mode_delta = 1e-3   #懒得改了，这里所有的delta都用22表示把
   omega = delta_w[:]
   omega[0] = omega[0]+mode_delta
   delta_w22_p = omega
   omega = delta_w[:]
   omega[0] = omega[0]-mode_delta
   delta_w22_m = omega
   deriv_w22 = (sf(freq,params_1_p,delta_w22_p,para_dtau) - sf(freq,params_1_m,delta_w22_m,para_dtau))/(2* mode_delta)


   mode_delta = 1e-3   #懒得改了，这里所有的delta都用22表示把
   omega = delta_w[:]
   omega[1] = omega[1]+mode_delta
   delta_w22_p = omega
   omega = delta_w[:]
   omega[1] = omega[1]-mode_delta
   delta_w22_m = omega
   deriv_w21 = (sf(freq,params_1_p,delta_w22_p,para_dtau) - sf(freq,params_1_m,delta_w22_m,para_dtau))/(2* mode_delta)


   mode_delta = 1e-3   #懒得改了，这里所有的delta都用22表示把
   omega = delta_w[:]
   omega[2] = omega[2]+mode_delta
   delta_w22_p = omega
   omega = delta_w[:]
   omega[2] = omega[2]-mode_delta
   delta_w22_m = omega
   deriv_w33 = (sf(freq,params_1_p,delta_w22_p,para_dtau) - sf(freq,params_1_m,delta_w22_m,para_dtau))/(2* mode_delta)

   mode_delta = 1e-3   #懒得改了，这里所有的delta都用22表示把
   omega = delta_w[:]
   omega[3] = omega[3]+mode_delta
   delta_w22_p = omega
   omega = delta_w[:]
   omega[3] = omega[3]-mode_delta
   delta_w22_m = omega
   deriv_w44 = (sf(freq,params_1_p,delta_w22_p,para_dtau) - sf(freq,params_1_m,delta_w22_m,para_dtau))/(2* mode_delta)

   #这里按顺序是tau的各模式
       #22模式的w和tau:



   mode_delta = 1e-3
   tau = delta_tau[:]
   tau[0] = tau[0]+mode_delta
   delta_tau22_p = tau
   tau = delta_tau[:]
   tau[0] = tau[0]-mode_delta
   delta_tau22_m = tau
   deriv_tau22 = (sf(freq,params_1_p,para_dw,delta_tau22_p) - sf(freq,params_1_m,para_dw,delta_tau22_m))/(2* mode_delta)


   #21模式的w和tau:




   mode_delta = 1e-3
   tau = delta_tau[:]
   tau[1] = tau[1]+mode_delta
   delta_tau22_p = tau
   tau = delta_tau[:]
   tau[1] = tau[1]-mode_delta
   delta_tau22_m = tau
   deriv_tau21 = (sf(freq,params_1_p,para_dw,delta_tau22_p) - sf(freq,params_1_m,para_dw,delta_tau22_m))/(2* mode_delta)

   #33模式的w和tau:



   mode_delta = 1e-3
   tau = delta_tau[:]
   tau[2] = tau[2]+mode_delta
   delta_tau22_p = tau
   tau = delta_tau[:]
   tau[2] = tau[2]-mode_delta
   delta_tau22_m = tau
   deriv_tau33 = (sf(freq,params_1_p,para_dw,delta_tau22_p) - sf(freq,params_1_m,para_dw,delta_tau22_m))/(2* mode_delta)

   #44模式的w和tau:



   mode_delta = 1e-3
   tau = delta_tau[:]
   tau[3] = tau[3]+mode_delta
   delta_tau22_p = tau
   tau = delta_tau[:]
   tau[3] = tau[3]-mode_delta
   delta_tau22_m = tau
   deriv_tau44 = (sf(freq,params_1_p,para_dw,delta_tau22_p) - sf(freq,params_1_m,para_dw,delta_tau22_m))/(2* mode_delta)







   diff_vec = [deriv_w22,deriv_w21,deriv_w33,deriv_w44,deriv_tau22,deriv_tau21,deriv_tau33,deriv_tau44]  # Concatenate derivatives

   N_sig = 1  # Number of signals

   N_params = len(diff_vec)  # Number of parameters we care about

   K = N_sig * N_params  # Dimension of Fisher Matrix

   fish_mix_tq = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         fish_mix_tq[i,j] = inner_prod(diff_vec[i],diff_vec[j],psd_tq(freq),1e-5)  # Construct Fisher Matrix
   fish_mix_ls = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         fish_mix_ls[i,j] = inner_prod(diff_vec[i],diff_vec[j],psd_ls(freq),1e-5)          
     

   import mpmath as mp  # Import arbitrary precision matrix
   #mp.dps表示精确度达到的位数
   mp.dps = 4000;   
   #print(fish_mix)
   fish_mix_prec = mp.matrix(fish_mix_tq)+mp.matrix(fish_mix_ls)
   #print(fish_mix_prec)
   fish_mix_inv = fish_mix_prec**-1
   #print(fish_mix_inv)

   #将这三行向量合成为一个矩阵
   Cov_Matrix = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         Cov_Matrix[i,j] = float(fish_mix_inv[i,j])
   print(Cov_Matrix)
   print('Delta logMchirp optimal:',np.sqrt(np.diag(Cov_Matrix))[0])
   print('Delta eta optimal:',np.sqrt(np.diag(Cov_Matrix))[1])
   print('Delta beta optimal:',np.sqrt(np.diag(Cov_Matrix))[2])
   print('Delta beta optimal:',np.sqrt(np.diag(Cov_Matrix))[3])
   return np.sqrt(np.diag(Cov_Matrix))[0],np.sqrt(np.diag(Cov_Matrix))[1],np.sqrt(np.diag(Cov_Matrix))[2],np.sqrt(np.diag(Cov_Matrix))[3],\
   np.sqrt(np.diag(Cov_Matrix))[4],np.sqrt(np.diag(Cov_Matrix))[5],np.sqrt(np.diag(Cov_Matrix))[6],np.sqrt(np.diag(Cov_Matrix))[7]