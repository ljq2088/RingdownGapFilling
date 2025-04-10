#这部分代码用于通过fisher矩阵计算quasinormal mode的标准差
#2024.8.28 china Beijing
###############################################################################
#这里是专门用于计算omega和tau的fisher矩阵，先计算的是最初的没有gap的fisher矩阵的结果#
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import dataprocess as dp
import waveform_eta as wf
import mpmath as mp

###############################################################################
#这里是专门用于计算omega和tau的fisher矩阵，先计算的是最初的没有gap的fisher矩阵的结果#
###############################################################################
#def fis_w(Mtot,M_ratio,R_shift,delta_w = [0,0,0,0],delta_tau=[0,0,0,0]):
def ls_fm(paramater,angles):

   Mtotw = 10**(paramater[0])
   spin = paramater[1]
   eta =0.45504534927350593+\
        1863.7071217428343 / (9.774183841831808e10 - 8.9603842815e10 * spin + 299339.0098450251 *np.sqrt(1.3997382578590552e11 - 1.9548367683663617e11 * spin+ 8.9603842815e10 * spin**2))**(1/3)\
        -0.00012938451784859424 * (9.774183841831808e10 - 8.9603842815e10 * spin + 299339.0098450251 * np.sqrt(1.3997382578590552e11 - 1.9548367683663617e11 * spin + 8.9603842815e10 * spin**2))**(1/3)
   M_ratiow = (1+np.sqrt(1-4*eta)-2*eta)/(2*eta)
   R_shiftw = paramater[2]

   delta_w = delta_tau = [0,0,0,0]
   para_dw = para_dtau = [0,0,0,0]
   params_1_p = params_1_m =paramater
   f_ini = wf.fin(Mtotw,M_ratiow)
   f_end = wf.fout(Mtotw,M_ratiow)
   freq = np.arange(f_ini,f_end,1e-5)
   mode_delta = 1e-6
   #params_1_p = [Mtotw,M_ratiow,R_shiftw]
   #params_1_m = [Mtotw,M_ratiow,R_shiftw]   
   #对log_mass的微分
   params_logmass1 = [paramater[0]+mode_delta,paramater[1],paramater[2]]
   params_logmass2 = [paramater[0]-mode_delta,paramater[1],paramater[2]]
   deriv_mass = (wf.sfa(freq,params_logmass1,angles,'lisa',delta_w,para_dtau) - wf.sfa(freq,params_logmass2,angles,'lisa',delta_w,para_dtau))/(2* mode_delta)

   #对对称质量比的微分
   params_spin1 = [paramater[0],paramater[1]+mode_delta,paramater[2]]
   params_spin2 = [paramater[0],paramater[1]-mode_delta,paramater[2]]
   deriv_spin = (wf.sfa(freq,params_spin1,angles,'lisa',delta_w,para_dtau) - wf.sfa(freq,params_spin2,angles,'lisa',delta_w,para_dtau))/(2* mode_delta)

   #omega 22 模式的微分
   delta_w22_p = delta_w[:]
   delta_w22_p[0] = mode_delta
   delta_w22_m = delta_w[:]
   delta_w22_m[0] = -mode_delta
   deriv_w22 = (wf.sfa(freq,params_1_p,angles,'lisa',delta_w22_p,para_dtau) - wf.sfa(freq,params_1_m,angles,'lisa',delta_w22_m,para_dtau))/(2* mode_delta)


   #omega 33 模式的微分
   delta_w33_p = delta_w[:]
   delta_w33_p[2] = mode_delta
   delta_w33_m = delta_w[:]
   delta_w33_m[2] = -mode_delta
   deriv_w33 = (wf.sfa(freq,params_1_p,angles,'lisa',delta_w33_p,para_dtau) - wf.sfa(freq,params_1_m,angles,'lisa',delta_w33_m,para_dtau))/(2* mode_delta)

   #omega 44 模式的微分
   delta_w44_p = delta_w[:]
   delta_w44_p[3] = mode_delta
   delta_w44_m = delta_w[:]
   delta_w44_m[3] = -mode_delta
   deriv_w44 = (wf.sfa(freq,params_1_p,angles,'lisa',delta_w44_p,para_dtau) - wf.sfa(freq,params_1_m,angles,'lisa',delta_w44_m,para_dtau))/(2* mode_delta)

   #tau 22 模式的微分
   delta_t22_p = delta_tau[:]
   delta_t22_p[0] = mode_delta
   delta_t22_m = delta_tau[:]
   delta_t22_m[0] = -mode_delta
   deriv_tau22 = (wf.sfa(freq,params_1_p,angles,'lisa',para_dw,delta_t22_p) - wf.sfa(freq,params_1_m,angles,'lisa',para_dw,delta_t22_m))/(2* mode_delta)



   #tau 33 模式的微分
   delta_t33_p = delta_tau[:]
   delta_t33_p[2] = mode_delta
   delta_t33_m = delta_tau[:]
   delta_t33_m[2] = -mode_delta
   deriv_tau33 = (wf.sfa(freq,params_1_p,angles,'lisa',para_dw,delta_t33_p) - wf.sfa(freq,params_1_m,angles,'lisa',para_dw,delta_t33_m))/(2* mode_delta)
   
   #tau 44 模式的微分
   delta_t44_p = delta_tau[:]
   delta_t44_p[3] = mode_delta
   delta_t44_m = delta_tau[:]
   delta_t44_m[3] = -mode_delta
   deriv_tau44 = (wf.sfa(freq,params_1_p,angles,'lisa',para_dw,delta_t44_p) - wf.sfa(freq,params_1_m,angles,'lisa',para_dw,delta_t44_m))/(2* mode_delta)



   diff_vec = [deriv_mass,deriv_spin,deriv_w22,deriv_w33,deriv_w44,deriv_tau22,deriv_tau33,deriv_tau44]  # Concatenate derivatives
   N_sig = 1  # Number of signals
   N_params = len(diff_vec)  # Number of parameters we care about
   K = N_sig * N_params  # Dimension of Fisher Matrix
   fish_mix = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         fish_mix[i,j] = dp.inner_prod(diff_vec[i],diff_vec[j],wf.PSD_ls(freq),1e-5)  # Construct Fisher Matrix

   import mpmath as mp  # Import arbitrary precision matrix
   mp.dps = 4000;     # mp.dps表示精确度达到的位数 
   fish_mix_prec = mp.matrix(fish_mix)
   fish_mix_inv = fish_mix_prec**-1

   #将这三行向量合成为一个矩阵
   Cov_Matrix = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         Cov_Matrix[i,j] = float(fish_mix_inv[i,j])
         
   return np.sqrt(np.diag(Cov_Matrix))[0],np.sqrt(np.diag(Cov_Matrix))[1],np.sqrt(np.diag(Cov_Matrix))[2],np.sqrt(np.diag(Cov_Matrix))[3],\
   np.sqrt(np.diag(Cov_Matrix))[4],np.sqrt(np.diag(Cov_Matrix))[5],np.sqrt(np.diag(Cov_Matrix))[6],np.sqrt(np.diag(Cov_Matrix))[7]


###############################################################################
#这里是专门用于计算omega和tau的fisher矩阵，先计算的是最初的没有gap的fisher矩阵的结果#
###############################################################################
#def fis_w(Mtot,M_ratio,R_shift,delta_w = [0,0,0,0],delta_tau=[0,0,0,0]):
def tj_fm(paramater,angles):

   Mtotw = 10**(paramater[0])
   spin = paramater[1]
   eta =0.45504534927350593+\
        1863.7071217428343 / (9.774183841831808e10 - 8.9603842815e10 * spin + 299339.0098450251 *np.sqrt(1.3997382578590552e11 - 1.9548367683663617e11 * spin+ 8.9603842815e10 * spin**2))**(1/3)\
        -0.00012938451784859424 * (9.774183841831808e10 - 8.9603842815e10 * spin + 299339.0098450251 * np.sqrt(1.3997382578590552e11 - 1.9548367683663617e11 * spin + 8.9603842815e10 * spin**2))**(1/3)
   M_ratiow = (1+np.sqrt(1-4*eta)-2*eta)/(2*eta)
   R_shiftw = paramater[2]

   delta_w = delta_tau = [0,0,0,0]
   para_dw = para_dtau = [0,0,0,0]
   params_1_p = params_1_m =paramater
   f_ini = wf.fin(Mtotw,M_ratiow)
   f_end = wf.fout(Mtotw,M_ratiow)
   freq = np.arange(f_ini,f_end,1e-5)
   mode_delta = 1e-6
   #params_1_p = [Mtotw,M_ratiow,R_shiftw]
   #params_1_m = [Mtotw,M_ratiow,R_shiftw]   

   params_logmass1 = [paramater[0]+mode_delta,paramater[1],paramater[2]]
   params_logmass2 = [paramater[0]-mode_delta,paramater[1],paramater[2]]
   deriv_mass = (wf.sfa(freq,params_logmass1,angles,'tj',delta_w,para_dtau) - wf.sfa(freq,params_logmass2,angles,'tj',delta_w,para_dtau))/(2* mode_delta)

   #对对称质量比的微分
   params_spin1 = [paramater[0],paramater[1]+mode_delta,paramater[2]]
   params_spin2 = [paramater[0],paramater[1]-mode_delta,paramater[2]]
   deriv_spin = (wf.sfa(freq,params_spin1,angles,'tj',delta_w,para_dtau) - wf.sfa(freq,params_spin2,angles,'tj',delta_w,para_dtau))/(2* mode_delta)

   #omega 22 模式的微分
   delta_w22_p = delta_w[:]
   delta_w22_p[0] = mode_delta
   delta_w22_m = delta_w[:]
   delta_w22_m[0] = -mode_delta
   deriv_w22 = (wf.sfa(freq,params_1_p,angles,'tj',delta_w22_p,para_dtau) - wf.sfa(freq,params_1_m,angles,'tj',delta_w22_m,para_dtau))/(2* mode_delta)

   #omega 33 模式的微分
   delta_w33_p = delta_w[:]
   delta_w33_p[2] = mode_delta
   delta_w33_m = delta_w[:]
   delta_w33_m[2] = -mode_delta
   deriv_w33 = (wf.sfa(freq,params_1_p,angles,'tj',delta_w33_p,para_dtau) - wf.sfa(freq,params_1_m,angles,'tj',delta_w33_m,para_dtau))/(2* mode_delta)

   #omega 44 模式的微分
   delta_w44_p = delta_w[:]
   delta_w44_p[3] = mode_delta
   delta_w44_m = delta_w[:]
   delta_w44_m[3] = -mode_delta
   deriv_w44 = (wf.sfa(freq,params_1_p,angles,'tj',delta_w44_p,para_dtau) - wf.sfa(freq,params_1_m,angles,'tj',delta_w44_m,para_dtau))/(2* mode_delta)

   #tau 22 模式的微分
   delta_t22_p = delta_tau[:]
   delta_t22_p[0] = mode_delta
   delta_t22_m = delta_tau[:]
   delta_t22_m[0] = -mode_delta
   deriv_tau22 = (wf.sfa(freq,params_1_p,angles,'tj',para_dw,delta_t22_p) - wf.sfa(freq,params_1_m,angles,'tj',para_dw,delta_t22_m))/(2* mode_delta)

   #tau 33 模式的微分
   delta_t33_p = delta_tau[:]
   delta_t33_p[2] = mode_delta
   delta_t33_m = delta_tau[:]
   delta_t33_m[2] = -mode_delta
   deriv_tau33 = (wf.sfa(freq,params_1_p,angles,'tj',para_dw,delta_t33_p) - wf.sfa(freq,params_1_m,angles,'tj',para_dw,delta_t33_m))/(2* mode_delta)
   
   #tau 44 模式的微分
   delta_t44_p = delta_tau[:]
   delta_t44_p[3] = mode_delta
   delta_t44_m = delta_tau[:]
   delta_t44_m[3] = -mode_delta
   deriv_tau44 = (wf.sfa(freq,params_1_p,angles,'tj',para_dw,delta_t44_p) - wf.sfa(freq,params_1_m,angles,'tj',para_dw,delta_t44_m))/(2* mode_delta)



   diff_vec = [deriv_mass,deriv_spin,deriv_w22,deriv_w33,deriv_w44,deriv_tau22,deriv_tau33,deriv_tau44]  # Concatenate derivatives
   N_sig = 1  # Number of signals
   N_params = len(diff_vec)  # Number of parameters we care about
   K = N_sig * N_params  # Dimension of Fisher Matrix
   fish_mix = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         fish_mix[i,j] = dp.inner_prod(diff_vec[i],diff_vec[j],wf.PSD_tj(freq),1e-5)  # Construct Fisher Matrix

   import mpmath as mp  # Import arbitrary precision matrix
   mp.dps = 4000;     # mp.dps表示精确度达到的位数 
   fish_mix_prec = mp.matrix(fish_mix)
   fish_mix_inv = fish_mix_prec**-1

   #将这三行向量合成为一个矩阵
   Cov_Matrix = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         Cov_Matrix[i,j] = float(fish_mix_inv[i,j])
         
   return np.sqrt(np.diag(Cov_Matrix))[0],np.sqrt(np.diag(Cov_Matrix))[1],np.sqrt(np.diag(Cov_Matrix))[2],np.sqrt(np.diag(Cov_Matrix))[3],\
   np.sqrt(np.diag(Cov_Matrix))[4],np.sqrt(np.diag(Cov_Matrix))[5],np.sqrt(np.diag(Cov_Matrix))[6],np.sqrt(np.diag(Cov_Matrix))[7]

def tq_fm(paramater,angles):

   Mtotw = 10**(paramater[0])
   spin = paramater[1]
   eta =0.45504534927350593+\
        1863.7071217428343 / (9.774183841831808e10 - 8.9603842815e10 * spin + 299339.0098450251 *np.sqrt(1.3997382578590552e11 - 1.9548367683663617e11 * spin+ 8.9603842815e10 * spin**2))**(1/3)\
        -0.00012938451784859424 * (9.774183841831808e10 - 8.9603842815e10 * spin + 299339.0098450251 * np.sqrt(1.3997382578590552e11 - 1.9548367683663617e11 * spin + 8.9603842815e10 * spin**2))**(1/3)
   M_ratiow = (1+np.sqrt(1-4*eta)-2*eta)/(2*eta)
   R_shiftw = paramater[2]

   delta_w = delta_tau = [0,0,0,0]
   para_dw = para_dtau = [0,0,0,0]
   params_1_p = params_1_m =paramater
   f_ini = wf.fin(Mtotw,M_ratiow)
   f_end = wf.fout(Mtotw,M_ratiow)
   freq = np.arange(f_ini,f_end,1e-5)
   mode_delta = 1e-6
   #params_1_p = [Mtotw,M_ratiow,R_shiftw]
   #params_1_m = [Mtotw,M_ratiow,R_shiftw]   
   #对log_mass的微分
   params_logmass1 = [paramater[0]+mode_delta,paramater[1],paramater[2]]
   params_logmass2 = [paramater[0]-mode_delta,paramater[1],paramater[2]]
   deriv_mass = (wf.sfa(freq,params_logmass1,angles,'tq',delta_w,para_dtau) - wf.sfa(freq,params_logmass2,angles,'tq',delta_w,para_dtau))/(2* mode_delta)

   #对对称质量比的微分
   params_spin1 = [paramater[0],paramater[1]+mode_delta,paramater[2]]
   params_spin2 = [paramater[0],paramater[1]-mode_delta,paramater[2]]
   deriv_spin = (wf.sfa(freq,params_spin1,angles,'tq',delta_w,para_dtau) - wf.sfa(freq,params_spin2,angles,'tq',delta_w,para_dtau))/(2* mode_delta)

 
   #omega 22 模式的微分
   delta_w22_p = delta_w[:]
   delta_w22_p[0] = mode_delta
   delta_w22_m = delta_w[:]
   delta_w22_m[0] = -mode_delta
   deriv_w22 = (wf.sfa(freq,params_1_p,angles,'tq',delta_w22_p,para_dtau) - wf.sfa(freq,params_1_m,angles,'tq',delta_w22_m,para_dtau))/(2* mode_delta)

   #omega 33 模式的微分
   delta_w33_p = delta_w[:]
   delta_w33_p[2] = mode_delta
   delta_w33_m = delta_w[:]
   delta_w33_m[2] = -mode_delta
   deriv_w33 = (wf.sfa(freq,params_1_p,angles,'tq',delta_w33_p,para_dtau) - wf.sfa(freq,params_1_m,angles,'tq',delta_w33_m,para_dtau))/(2* mode_delta)

   #omega 44 模式的微分
   delta_w44_p = delta_w[:]
   delta_w44_p[3] = mode_delta
   delta_w44_m = delta_w[:]
   delta_w44_m[3] = -mode_delta
   deriv_w44 = (wf.sfa(freq,params_1_p,angles,'tq',delta_w44_p,para_dtau) - wf.sfa(freq,params_1_m,angles,'tq',delta_w44_m,para_dtau))/(2* mode_delta)

   #tau 22 模式的微分
   delta_t22_p = delta_tau[:]
   delta_t22_p[0] = mode_delta
   delta_t22_m = delta_tau[:]
   delta_t22_m[0] = -mode_delta
   deriv_tau22 = (wf.sfa(freq,params_1_p,angles,'tq',para_dw,delta_t22_p) - wf.sfa(freq,params_1_m,angles,'tq',para_dw,delta_t22_m))/(2* mode_delta)

   #tau 33 模式的微分
   delta_t33_p = delta_tau[:]
   delta_t33_p[2] = mode_delta
   delta_t33_m = delta_tau[:]
   delta_t33_m[2] = -mode_delta
   deriv_tau33 = (wf.sfa(freq,params_1_p,angles,'tq',para_dw,delta_t33_p) - wf.sfa(freq,params_1_m,angles,'tq',para_dw,delta_t33_m))/(2* mode_delta)
   
   #tau 44 模式的微分
   delta_t44_p = delta_tau[:]
   delta_t44_p[3] = mode_delta
   delta_t44_m = delta_tau[:]
   delta_t44_m[3] = -mode_delta
   deriv_tau44 = (wf.sfa(freq,params_1_p,angles,'tq',para_dw,delta_t44_p) - wf.sfa(freq,params_1_m,angles,'tq',para_dw,delta_t44_m))/(2* mode_delta)



   diff_vec = [deriv_mass,deriv_spin,deriv_w22,deriv_w33,deriv_w44,deriv_tau22,deriv_tau33,deriv_tau44]  # Concatenate derivatives
   N_sig = 1  # Number of signals
   N_params = len(diff_vec)  # Number of parameters we care about
   K = N_sig * N_params  # Dimension of Fisher Matrix
   fish_mix = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         fish_mix[i,j] = dp.inner_prod(diff_vec[i],diff_vec[j],wf.PSD_tq(freq),1e-5)  # Construct Fisher Matrix

   import mpmath as mp  # Import arbitrary precision matrix
   mp.dps = 4000;     # mp.dps表示精确度达到的位数 
   fish_mix_prec = mp.matrix(fish_mix)
   fish_mix_inv = fish_mix_prec**-1

   #将这三行向量合成为一个矩阵
   Cov_Matrix = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         Cov_Matrix[i,j] = float(fish_mix_inv[i,j])
         
   return np.sqrt(np.diag(Cov_Matrix))[0],np.sqrt(np.diag(Cov_Matrix))[1],np.sqrt(np.diag(Cov_Matrix))[2],np.sqrt(np.diag(Cov_Matrix))[3],\
   np.sqrt(np.diag(Cov_Matrix))[4],np.sqrt(np.diag(Cov_Matrix))[5],np.sqrt(np.diag(Cov_Matrix))[6],np.sqrt(np.diag(Cov_Matrix))[7]


######################################################################################################################################
###########################################计算带有75%占空比的数据间隙数据的Fisher矩阵###################################################
######################################################################################################################################
def gp_ls_fm(freq_arr,paramater,angles,random,detector = 'lisa'):
   #freq_arr：总的探测频段的频谱数组
   #paramater，angles：黑洞参数和角度参数
   #random：gap相对ringdown出现的位置，取值在[0,1]
   #detector：使用的探测器，默认就是LISA
   Mtotw = 10**(paramater[0])
   spin = paramater[1]
   eta =0.45504534927350593+\
        1863.7071217428343 / (9.774183841831808e10 - 8.9603842815e10 * spin + 299339.0098450251 *np.sqrt(1.3997382578590552e11 - 1.9548367683663617e11 * spin+ 8.9603842815e10 * spin**2))**(1/3)\
        -0.00012938451784859424 * (9.774183841831808e10 - 8.9603842815e10 * spin + 299339.0098450251 * np.sqrt(1.3997382578590552e11 - 1.9548367683663617e11 * spin + 8.9603842815e10 * spin**2))**(1/3)
   M_ratiow = (1+np.sqrt(1-4*eta)-2*eta)/(2*eta)
   R_shiftw = paramater[2]
   nn = 0
   while freq_arr[nn]<wf.fin(Mtotw,M_ratiow):
      nn+=1
   n_s = nn
   while freq_arr[nn]<wf.fout(Mtotw,M_ratiow):
      nn+=1
   n_e = nn
   freq_rd_gap = freq_arr[n_s:n_e]

   delta_w = delta_tau = [0,0,0,0]
   para_dw = para_dtau = [0,0,0,0]
  
   freq = freq_arr
   mode_delta = 1e-3
   #params_1_p = [Mtotw,M_ratiow,R_shiftw]
   #params_1_m = [Mtotw,M_ratiow,R_shiftw]   

   #对log_mass的微分
   params_logmass1 = [paramater[0]+mode_delta,paramater[1],paramater[2]]
   params_logmass2 = [paramater[0]-mode_delta,paramater[1],paramater[2]]
   deriv_mass = (wf.gap_sfa_fm(freq,params_logmass1,angles,n_s,n_e,random,detector,delta_w,para_dtau) -\
                  wf.gap_sfa_fm(freq,params_logmass2,angles,n_s,n_e,random,detector,delta_w,para_dtau))/(2* mode_delta)

   #对对称质量比的微分
   params_spin1 = [paramater[0],paramater[1]+mode_delta,paramater[2]]
   params_spin2 = [paramater[0],paramater[1]-mode_delta,paramater[2]]
   deriv_spin = (wf.gap_sfa_fm(freq,params_spin1,angles,n_s,n_e,random,detector,delta_w,para_dtau) -\
                  wf.gap_sfa_fm(freq,params_spin2,angles,n_s,n_e,random,detector,delta_w,para_dtau))/(2* mode_delta)


   #omega 22 模式的微分
   delta_w22_p = delta_w[:]
   delta_w22_p[0] = mode_delta
   delta_w22_m = delta_w[:]
   delta_w22_m[0] = -mode_delta
   deriv_w22 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w22_p,para_dtau)-\
                wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w22_m,para_dtau))/(2* mode_delta)


   #omega 33 模式的微分
   delta_w33_p = delta_w[:]
   delta_w33_p[2] = mode_delta
   delta_w33_m = delta_w[:]
   delta_w33_m[2] = -mode_delta
   deriv_w33 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w33_p,para_dtau)-\
                wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w33_m,para_dtau))/(2* mode_delta)

   #omega 44 模式的微分
   delta_w44_p = delta_w[:]
   delta_w44_p[3] = mode_delta
   delta_w44_m = delta_w[:]
   delta_w44_m[3] = -mode_delta
   deriv_w44 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w44_p,para_dtau)-\
                wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w44_m,para_dtau))/(2* mode_delta)

   #tau 22 模式的微分
   delta_t22_p = delta_tau[:]
   delta_t22_p[0] = mode_delta
   delta_t22_m = delta_tau[:]
   delta_t22_m[0] = -mode_delta
   deriv_tau22 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t22_p) - \
                  wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t22_m))/(2* mode_delta)


   #tau 33 模式的微分
   delta_t33_p = delta_tau[:]
   delta_t33_p[2] = mode_delta
   delta_t33_m = delta_tau[:]
   delta_t33_m[2] = -mode_delta
   deriv_tau33 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t33_p) - \
                  wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t33_m))/(2* mode_delta)
   
   #tau 44 模式的微分
   delta_t44_p = delta_tau[:]
   delta_t44_p[3] = mode_delta
   delta_t44_m = delta_tau[:]
   delta_t44_m[3] = -mode_delta
   deriv_tau44 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t44_p) - \
                  wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t44_m))/(2* mode_delta)



   diff_vec = [deriv_mass,deriv_spin,deriv_w22,deriv_w33,deriv_w44,deriv_tau22,deriv_tau33,deriv_tau44]  # Concatenate derivatives
   N_sig = 1  # Number of signals
   N_params = len(diff_vec)  # Number of parameters we care about
   K = N_sig * N_params  # Dimension of Fisher Matrix
   delta_freqency = freq_rd_gap[1]-freq_rd_gap[0] #在计算内积时需要的频率间隔
   fish_mix = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         fish_mix[i,j] = dp.inner_prod(diff_vec[i],diff_vec[j],wf.PSD_ls(freq_rd_gap),delta_freqency)  # Construct Fisher Matrix

   import mpmath as mp  # Import arbitrary precision matrix
   mp.dps = 4000;     # mp.dps表示精确度达到的位数 
   fish_mix_prec = mp.matrix(fish_mix)
   fish_mix_inv = fish_mix_prec**-1

   #将这三行向量合成为一个矩阵
   Cov_Matrix = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         Cov_Matrix[i,j] = float(fish_mix_inv[i,j])
         
   return np.sqrt(np.diag(Cov_Matrix))[0],np.sqrt(np.diag(Cov_Matrix))[1],np.sqrt(np.diag(Cov_Matrix))[2],np.sqrt(np.diag(Cov_Matrix))[3],\
   np.sqrt(np.diag(Cov_Matrix))[4],np.sqrt(np.diag(Cov_Matrix))[5],np.sqrt(np.diag(Cov_Matrix))[6],np.sqrt(np.diag(Cov_Matrix))[7]

def gp_tj_fm(freq_arr,paramater,angles,random,detector = 'tj'):
   #freq_arr：总的探测频段的频谱数组
   #paramater，angles：黑洞参数和角度参数
   #random：gap相对ringdown出现的位置，取值在[0,1]
   #detector：使用的探测器，默认就是LISA
   Mtotw = paramater[0]
   M_ratiow = paramater[1]
   R_shiftw = paramater[2]
   nn = 0
   while freq_arr[nn]<wf.fin(Mtotw,M_ratiow):
      nn+=1
   n_s = nn
   while freq_arr[nn]<wf.fout(Mtotw,M_ratiow):
      nn+=1
   n_e = nn
   freq_rd_gap = freq_arr[n_s:n_e]

   delta_w = delta_tau = [0,0,0,0]
   para_dw = para_dtau = [0,0,0,0]
   params_1_p = params_1_m =paramater
   freq = freq_arr
   mode_delta = 1e-3
   #params_1_p = [Mtotw,M_ratiow,R_shiftw]
   #params_1_m = [Mtotw,M_ratiow,R_shiftw]   

   #omega 22 模式的微分
   delta_w22_p = delta_w[:]
   delta_w22_p[0] = mode_delta
   delta_w22_m = delta_w[:]
   delta_w22_m[0] = -mode_delta
   deriv_w22 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w22_p,para_dtau)-\
                wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w22_m,para_dtau))/(2* mode_delta)


   #omega 21 模式的微分
   delta_w21_p = delta_w[:]
   delta_w21_p[1] = mode_delta
   delta_w21_m = delta_w[:]
   delta_w21_m[1] = -mode_delta
   deriv_w21 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w21_p,para_dtau)-\
                wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w21_m,para_dtau))/(2* mode_delta)

   #omega 33 模式的微分
   delta_w33_p = delta_w[:]
   delta_w33_p[2] = mode_delta
   delta_w33_m = delta_w[:]
   delta_w33_m[2] = -mode_delta
   deriv_w33 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w33_p,para_dtau)-\
                wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w33_m,para_dtau))/(2* mode_delta)

   #omega 44 模式的微分
   delta_w44_p = delta_w[:]
   delta_w44_p[3] = mode_delta
   delta_w44_m = delta_w[:]
   delta_w44_m[3] = -mode_delta
   deriv_w44 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w44_p,para_dtau)-\
                wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w44_m,para_dtau))/(2* mode_delta)

   #tau 22 模式的微分
   delta_t22_p = delta_tau[:]
   delta_t22_p[0] = mode_delta
   delta_t22_m = delta_tau[:]
   delta_t22_m[0] = -mode_delta
   deriv_tau22 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t22_p) - \
                  wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t22_m))/(2* mode_delta)


   #tau 21 模式的微分
   delta_t21_p = delta_tau[:]
   delta_t21_p[1] = mode_delta
   delta_t21_m = delta_tau[:]
   delta_t21_m[1] = -mode_delta
   deriv_tau21 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t21_p) - \
                  wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t21_m))/(2* mode_delta)

   #tau 33 模式的微分
   delta_t33_p = delta_tau[:]
   delta_t33_p[2] = mode_delta
   delta_t33_m = delta_tau[:]
   delta_t33_m[2] = -mode_delta
   deriv_tau33 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t33_p) - \
                  wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t33_m))/(2* mode_delta)
   
   #tau 44 模式的微分
   delta_t44_p = delta_tau[:]
   delta_t44_p[3] = mode_delta
   delta_t44_m = delta_tau[:]
   delta_t44_m[3] = -mode_delta
   deriv_tau44 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t44_p) - \
                  wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t44_m))/(2* mode_delta)



   diff_vec = [deriv_w22,deriv_w21,deriv_w33,deriv_w44,deriv_tau22,deriv_tau21,deriv_tau33,deriv_tau44]  # Concatenate derivatives
   N_sig = 1  # Number of signals
   N_params = len(diff_vec)  # Number of parameters we care about
   K = N_sig * N_params  # Dimension of Fisher Matrix
   delta_freqency = freq_rd_gap[1]-freq_rd_gap[0] #在计算内积时需要的频率间隔
   fish_mix = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         fish_mix[i,j] = dp.inner_prod(diff_vec[i],diff_vec[j],wf.PSD_tj(freq_rd_gap),delta_freqency)  # Construct Fisher Matrix

   import mpmath as mp  # Import arbitrary precision matrix
   mp.dps = 4000;     # mp.dps表示精确度达到的位数 
   fish_mix_prec = mp.matrix(fish_mix)
   fish_mix_inv = fish_mix_prec**-1

   #将这三行向量合成为一个矩阵
   Cov_Matrix = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         Cov_Matrix[i,j] = float(fish_mix_inv[i,j])
         
   return np.sqrt(np.diag(Cov_Matrix))[0],np.sqrt(np.diag(Cov_Matrix))[1],np.sqrt(np.diag(Cov_Matrix))[2],np.sqrt(np.diag(Cov_Matrix))[3],\
   np.sqrt(np.diag(Cov_Matrix))[4],np.sqrt(np.diag(Cov_Matrix))[5],np.sqrt(np.diag(Cov_Matrix))[6],np.sqrt(np.diag(Cov_Matrix))[7]

def gp_tq_fm(freq_arr,paramater,angles,random,detector = 'tq'):
   #freq_arr：总的探测频段的频谱数组
   #paramater，angles：黑洞参数和角度参数
   #random：gap相对ringdown出现的位置，取值在[0,1]
   #detector：使用的探测器，默认就是LISA
   Mtotw = paramater[0]
   M_ratiow = paramater[1]
   R_shiftw = paramater[2]
   nn = 0
   while freq_arr[nn]<wf.fin(Mtotw,M_ratiow):
      nn+=1
   n_s = nn
   while freq_arr[nn]<wf.fout(Mtotw,M_ratiow):
      nn+=1
   n_e = nn
   freq_rd_gap = freq_arr[n_s:n_e]

   delta_w = delta_tau = [0,0,0,0]
   para_dw = para_dtau = [0,0,0,0]
   params_1_p = params_1_m =paramater
   freq = freq_arr
   mode_delta = 1e-3
   #params_1_p = [Mtotw,M_ratiow,R_shiftw]
   #params_1_m = [Mtotw,M_ratiow,R_shiftw]   

   #omega 22 模式的微分
   delta_w22_p = delta_w[:]
   delta_w22_p[0] = mode_delta
   delta_w22_m = delta_w[:]
   delta_w22_m[0] = -mode_delta
   deriv_w22 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w22_p,para_dtau)-\
                wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w22_m,para_dtau))/(2* mode_delta)


   #omega 21 模式的微分
   delta_w21_p = delta_w[:]
   delta_w21_p[1] = mode_delta
   delta_w21_m = delta_w[:]
   delta_w21_m[1] = -mode_delta
   deriv_w21 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w21_p,para_dtau)-\
                wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w21_m,para_dtau))/(2* mode_delta)

   #omega 33 模式的微分
   delta_w33_p = delta_w[:]
   delta_w33_p[2] = mode_delta
   delta_w33_m = delta_w[:]
   delta_w33_m[2] = -mode_delta
   deriv_w33 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w33_p,para_dtau)-\
                wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w33_m,para_dtau))/(2* mode_delta)

   #omega 44 模式的微分
   delta_w44_p = delta_w[:]
   delta_w44_p[3] = mode_delta
   delta_w44_m = delta_w[:]
   delta_w44_m[3] = -mode_delta
   deriv_w44 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w44_p,para_dtau)-\
                wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,delta_w44_m,para_dtau))/(2* mode_delta)

   #tau 22 模式的微分
   delta_t22_p = delta_tau[:]
   delta_t22_p[0] = mode_delta
   delta_t22_m = delta_tau[:]
   delta_t22_m[0] = -mode_delta
   deriv_tau22 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t22_p) - \
                  wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t22_m))/(2* mode_delta)


   #tau 21 模式的微分
   delta_t21_p = delta_tau[:]
   delta_t21_p[1] = mode_delta
   delta_t21_m = delta_tau[:]
   delta_t21_m[1] = -mode_delta
   deriv_tau21 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t21_p) - \
                  wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t21_m))/(2* mode_delta)

   #tau 33 模式的微分
   delta_t33_p = delta_tau[:]
   delta_t33_p[2] = mode_delta
   delta_t33_m = delta_tau[:]
   delta_t33_m[2] = -mode_delta
   deriv_tau33 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t33_p) - \
                  wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t33_m))/(2* mode_delta)
   
   #tau 44 模式的微分
   delta_t44_p = delta_tau[:]
   delta_t44_p[3] = mode_delta
   delta_t44_m = delta_tau[:]
   delta_t44_m[3] = -mode_delta
   deriv_tau44 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t44_p) - \
                  wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random,detector,para_dw,delta_t44_m))/(2* mode_delta)



   diff_vec = [deriv_w22,deriv_w21,deriv_w33,deriv_w44,deriv_tau22,deriv_tau21,deriv_tau33,deriv_tau44]  # Concatenate derivatives
   N_sig = 1  # Number of signals
   N_params = len(diff_vec)  # Number of parameters we care about
   K = N_sig * N_params  # Dimension of Fisher Matrix
   delta_freqency = freq_rd_gap[1]-freq_rd_gap[0] #在计算内积时需要的频率间隔
   fish_mix = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         fish_mix[i,j] = dp.inner_prod(diff_vec[i],diff_vec[j],wf.PSD_tq(freq_rd_gap),delta_freqency)  # Construct Fisher Matrix

   import mpmath as mp  # Import arbitrary precision matrix
   mp.dps = 4000;     # mp.dps表示精确度达到的位数 
   fish_mix_prec = mp.matrix(fish_mix)
   fish_mix_inv = fish_mix_prec**-1

   #将这三行向量合成为一个矩阵
   Cov_Matrix = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         Cov_Matrix[i,j] = float(fish_mix_inv[i,j])
         
   return np.sqrt(np.diag(Cov_Matrix))[0],np.sqrt(np.diag(Cov_Matrix))[1],np.sqrt(np.diag(Cov_Matrix))[2],np.sqrt(np.diag(Cov_Matrix))[3],\
   np.sqrt(np.diag(Cov_Matrix))[4],np.sqrt(np.diag(Cov_Matrix))[5],np.sqrt(np.diag(Cov_Matrix))[6],np.sqrt(np.diag(Cov_Matrix))[7]

#两个探测器分别默认为：第一个是太极，第二个是天琴
def comb_fm(paramater,angles,detector1 = 'tj',detector2 = 'tq'):
   
   Mtotw = paramater[0]
   M_ratiow = paramater[1]
   R_shiftw = paramater[2]

   delta_w = delta_tau = [0,0,0,0]
   para_dw = para_dtau = [0,0,0,0]
   params_1_p = params_1_m = paramater
   f_ini = wf.fin(Mtotw,M_ratiow)
   f_end = wf.fout(Mtotw,M_ratiow)
   freq = np.arange(f_ini,f_end,1e-5)
   mode_delta = 1e-3
   #params_1_p = [Mtotw,M_ratiow,R_shiftw]
   #params_1_m = [Mtotw,M_ratiow,R_shiftw]   

   #omega 22 模式的微分
   delta_w22_p = delta_w[:]
   delta_w22_p[0] = mode_delta
   delta_w22_m = delta_w[:]
   delta_w22_m[0] = -mode_delta
   deriv_w22 = (wf.sfa(freq,params_1_p,angles,detector1,delta_w22_p,para_dtau) - wf.sfa(freq,params_1_m,angles,detector1,delta_w22_m,para_dtau))/(2* mode_delta)

   #omega 21 模式的微分
   delta_w21_p = delta_w[:]
   delta_w21_p[1] = mode_delta
   delta_w21_m = delta_w[:]
   delta_w21_m[1] = -mode_delta
   deriv_w21 = (wf.sfa(freq,params_1_p,angles,detector1,delta_w21_p,para_dtau) - wf.sfa(freq,params_1_m,angles,detector1,delta_w21_m,para_dtau))/(2* mode_delta)

   #omega 33 模式的微分
   delta_w33_p = delta_w[:]
   delta_w33_p[2] = mode_delta
   delta_w33_m = delta_w[:]
   delta_w33_m[2] = -mode_delta
   deriv_w33 = (wf.sfa(freq,params_1_p,angles,detector1,delta_w33_p,para_dtau) - wf.sfa(freq,params_1_m,angles,detector1,delta_w33_m,para_dtau))/(2* mode_delta)

   #omega 44 模式的微分
   delta_w44_p = delta_w[:]
   delta_w44_p[3] = mode_delta
   delta_w44_m = delta_w[:]
   delta_w44_m[3] = -mode_delta
   deriv_w44 = (wf.sfa(freq,params_1_p,angles,detector1,delta_w44_p,para_dtau) - wf.sfa(freq,params_1_m,angles,detector1,delta_w44_m,para_dtau))/(2* mode_delta)

   #tau 22 模式的微分
   delta_t22_p = delta_tau[:]
   delta_t22_p[0] = mode_delta
   delta_t22_m = delta_tau[:]
   delta_t22_m[0] = -mode_delta
   deriv_tau22 = (wf.sfa(freq,params_1_p,angles,detector1,para_dw,delta_t22_p) - wf.sfa(freq,params_1_m,angles,detector1,para_dw,delta_t22_m))/(2* mode_delta)


   #tau 21 模式的微分
   delta_t21_p = delta_tau[:]
   delta_t21_p[1] = mode_delta
   delta_t21_m = delta_tau[:]
   delta_t21_m[1] = -mode_delta
   deriv_tau21 = (wf.sfa(freq,params_1_p,angles,detector1,para_dw,delta_t21_p) - wf.sfa(freq,params_1_m,angles,detector1,para_dw,delta_t21_m))/(2* mode_delta)

   #tau 33 模式的微分
   delta_t33_p = delta_tau[:]
   delta_t33_p[2] = mode_delta
   delta_t33_m = delta_tau[:]
   delta_t33_m[2] = -mode_delta
   deriv_tau33 = (wf.sfa(freq,params_1_p,angles,detector1,para_dw,delta_t33_p) - wf.sfa(freq,params_1_m,angles,detector1,para_dw,delta_t33_m))/(2* mode_delta)
   
   #tau 44 模式的微分
   delta_t44_p = delta_tau[:]
   delta_t44_p[3] = mode_delta
   delta_t44_m = delta_tau[:]
   delta_t44_m[3] = -mode_delta
   deriv_tau44 = (wf.sfa(freq,params_1_p,angles,detector1,para_dw,delta_t44_p) - wf.sfa(freq,params_1_m,angles,detector1,para_dw,delta_t44_m))/(2* mode_delta)



   diff_vec1 = [deriv_w22,deriv_w21,deriv_w33,deriv_w44,deriv_tau22,deriv_tau21,deriv_tau33,deriv_tau44]  # Concatenate derivatives
   N_sig = 1  # Number of signals
   N_params = len(diff_vec1)  # Number of parameters we care about
   K = N_sig * N_params  # Dimension of Fisher Matrix
   fish_mix1 = np.eye(K)
   if detector1 == 'ls':
      for i in range(0,K):
         for j in range(0,K):
            fish_mix1[i,j] = dp.inner_prod(diff_vec1[i],diff_vec1[j],wf.PSD_ls(freq),1e-5)  # Construct Fisher Matrix
   elif detector1 == 'tq':
      for i in range(0,K):
         for j in range(0,K):
            fish_mix1[i,j] = dp.inner_prod(diff_vec1[i],diff_vec1[j],wf.PSD_tq(freq),1e-5)  # Construct Fisher Matrix
   else:
      for i in range(0,K):
         for j in range(0,K):
            fish_mix1[i,j] = dp.inner_prod(diff_vec1[i],diff_vec1[j],wf.PSD_tj(freq),1e-5)  # Construct Fisher Matrix


   delta_w22_p = delta_w[:]
   delta_w22_p[0] = mode_delta
   delta_w22_m = delta_w[:]
   delta_w22_m[0] = -mode_delta
   deriv_w22 = (wf.sfa(freq,params_1_p,angles,detector2,delta_w22_p,para_dtau) - wf.sfa(freq,params_1_m,angles,detector2,delta_w22_m,para_dtau))/(2* mode_delta)

   #omega 21 模式的微分
   delta_w21_p = delta_w[:]
   delta_w21_p[1] = mode_delta
   delta_w21_m = delta_w[:]
   delta_w21_m[1] = -mode_delta
   deriv_w21 = (wf.sfa(freq,params_1_p,angles,detector2,delta_w21_p,para_dtau) - wf.sfa(freq,params_1_m,angles,detector2,delta_w21_m,para_dtau))/(2* mode_delta)

   #omega 33 模式的微分
   delta_w33_p = delta_w[:]
   delta_w33_p[2] = mode_delta
   delta_w33_m = delta_w[:]
   delta_w33_m[2] = -mode_delta
   deriv_w33 = (wf.sfa(freq,params_1_p,angles,detector2,delta_w33_p,para_dtau) - wf.sfa(freq,params_1_m,angles,detector2,delta_w33_m,para_dtau))/(2* mode_delta)

   #omega 44 模式的微分
   delta_w44_p = delta_w[:]
   delta_w44_p[3] = mode_delta
   delta_w44_m = delta_w[:]
   delta_w44_m[3] = -mode_delta
   deriv_w44 = (wf.sfa(freq,params_1_p,angles,detector2,delta_w44_p,para_dtau) - wf.sfa(freq,params_1_m,angles,detector2,delta_w44_m,para_dtau))/(2* mode_delta)

   #tau 22 模式的微分
   delta_t22_p = delta_tau[:]
   delta_t22_p[0] = mode_delta
   delta_t22_m = delta_tau[:]
   delta_t22_m[0] = -mode_delta
   deriv_tau22 = (wf.sfa(freq,params_1_p,angles,detector2,para_dw,delta_t22_p) - wf.sfa(freq,params_1_m,angles,detector2,para_dw,delta_t22_m))/(2* mode_delta)


   #tau 21 模式的微分
   delta_t21_p = delta_tau[:]
   delta_t21_p[1] = mode_delta
   delta_t21_m = delta_tau[:]
   delta_t21_m[1] = -mode_delta
   deriv_tau21 = (wf.sfa(freq,params_1_p,angles,detector2,para_dw,delta_t21_p) - wf.sfa(freq,params_1_m,angles,detector2,para_dw,delta_t21_m))/(2* mode_delta)

   #tau 33 模式的微分
   delta_t33_p = delta_tau[:]
   delta_t33_p[2] = mode_delta
   delta_t33_m = delta_tau[:]
   delta_t33_m[2] = -mode_delta
   deriv_tau33 = (wf.sfa(freq,params_1_p,angles,detector2,para_dw,delta_t33_p) - wf.sfa(freq,params_1_m,angles,detector2,para_dw,delta_t33_m))/(2* mode_delta)
   
   #tau 44 模式的微分
   delta_t44_p = delta_tau[:]
   delta_t44_p[3] = mode_delta
   delta_t44_m = delta_tau[:]
   delta_t44_m[3] = -mode_delta
   deriv_tau44 = (wf.sfa(freq,params_1_p,angles,detector2,para_dw,delta_t44_p) - wf.sfa(freq,params_1_m,angles,detector2,para_dw,delta_t44_m))/(2* mode_delta)



   diff_vec2 = [deriv_w22,deriv_w21,deriv_w33,deriv_w44,deriv_tau22,deriv_tau21,deriv_tau33,deriv_tau44]  # Concatenate derivatives
   N_sig = 1  # Number of signals
   N_params = len(diff_vec2)  # Number of parameters we care about
   K = N_sig * N_params  # Dimension of Fisher Matrix
   fish_mix2 = np.eye(K)
   if detector2 == 'ls':
      for i in range(0,K):
         for j in range(0,K):
            fish_mix2[i,j] = dp.inner_prod(diff_vec2[i],diff_vec2[j],wf.PSD_ls(freq),1e-5)  # Construct Fisher Matrix
   elif detector2 == 'tj':
      for i in range(0,K):
         for j in range(0,K):
            fish_mix2[i,j] = dp.inner_prod(diff_vec2[i],diff_vec2[j],wf.PSD_tj(freq),1e-5)  # Construct Fisher Matrix
   else:
      for i in range(0,K):
         for j in range(0,K):
            fish_mix2[i,j] = dp.inner_prod(diff_vec2[i],diff_vec2[j],wf.PSD_tq(freq),1e-5)  # Construct Fisher Matrix


  # Import arbitrary precision matrix
   mp.dps = 4000;     # mp.dps表示精确度达到的位数 
   fish_mix_prec = mp.matrix(fish_mix1)+mp.matrix(fish_mix2)
   fish_mix_inv = fish_mix_prec**-1

   #将这三行向量合成为一个矩阵
   Cov_Matrix = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         Cov_Matrix[i,j] = float(fish_mix_inv[i,j])
         
   return np.sqrt(np.diag(Cov_Matrix))[0],np.sqrt(np.diag(Cov_Matrix))[1],np.sqrt(np.diag(Cov_Matrix))[2],np.sqrt(np.diag(Cov_Matrix))[3],\
   np.sqrt(np.diag(Cov_Matrix))[4],np.sqrt(np.diag(Cov_Matrix))[5],np.sqrt(np.diag(Cov_Matrix))[6],np.sqrt(np.diag(Cov_Matrix))[7]


def gp_comb_fm(freq_arr,paramater,angles,random1 = 0.5,random2 = 0.5,detector1 = 'tj',detector2 = 'tq'):
   #freq_arr：总的探测频段的频谱数组
   #paramater，angles：黑洞参数和角度参数
   #random：gap相对ringdown出现的位置，取值在[0,1]
   #detector：使用的探测器，默认就是LISA
   Mtotw = paramater[0]
   M_ratiow = paramater[1]
   R_shiftw = paramater[2]
   nn = 0
   while freq_arr[nn]<wf.fin(Mtotw,M_ratiow):
      nn+=1
   n_s = nn
   while freq_arr[nn]<wf.fout(Mtotw,M_ratiow):
      nn+=1
   n_e = nn
   freq_rd_gap = freq_arr[n_s:n_e]

   delta_w = delta_tau = [0,0,0,0]
   para_dw = para_dtau = [0,0,0,0]
   params_1_p = params_1_m =paramater
   freq = freq_arr
   mode_delta = 1e-3
   #params_1_p = [Mtotw,M_ratiow,R_shiftw]
   #params_1_m = [Mtotw,M_ratiow,R_shiftw]   

   #omega 22 模式的微分
   delta_w22_p = delta_w[:]
   delta_w22_p[0] = mode_delta
   delta_w22_m = delta_w[:]
   delta_w22_m[0] = -mode_delta
   deriv_w22 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random1,detector1,delta_w22_p,para_dtau)-\
                wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random1,detector1,delta_w22_m,para_dtau))/(2* mode_delta)


   #omega 21 模式的微分
   delta_w21_p = delta_w[:]
   delta_w21_p[1] = mode_delta
   delta_w21_m = delta_w[:]
   delta_w21_m[1] = -mode_delta
   deriv_w21 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random1,detector1,delta_w21_p,para_dtau)-\
                wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random1,detector1,delta_w21_m,para_dtau))/(2* mode_delta)

   #omega 33 模式的微分
   delta_w33_p = delta_w[:]
   delta_w33_p[2] = mode_delta
   delta_w33_m = delta_w[:]
   delta_w33_m[2] = -mode_delta
   deriv_w33 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random1,detector1,delta_w33_p,para_dtau)-\
                wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random1,detector1,delta_w33_m,para_dtau))/(2* mode_delta)

   #omega 44 模式的微分
   delta_w44_p = delta_w[:]
   delta_w44_p[3] = mode_delta
   delta_w44_m = delta_w[:]
   delta_w44_m[3] = -mode_delta
   deriv_w44 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random1,detector1,delta_w44_p,para_dtau)-\
                wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random1,detector1,delta_w44_m,para_dtau))/(2* mode_delta)

   #tau 22 模式的微分
   delta_t22_p = delta_tau[:]
   delta_t22_p[0] = mode_delta
   delta_t22_m = delta_tau[:]
   delta_t22_m[0] = -mode_delta
   deriv_tau22 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random1,detector1,para_dw,delta_t22_p) - \
                  wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random1,detector1,para_dw,delta_t22_m))/(2* mode_delta)


   #tau 21 模式的微分
   delta_t21_p = delta_tau[:]
   delta_t21_p[1] = mode_delta
   delta_t21_m = delta_tau[:]
   delta_t21_m[1] = -mode_delta
   deriv_tau21 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random1,detector1,para_dw,delta_t21_p) - \
                  wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random1,detector1,para_dw,delta_t21_m))/(2* mode_delta)

   #tau 33 模式的微分
   delta_t33_p = delta_tau[:]
   delta_t33_p[2] = mode_delta
   delta_t33_m = delta_tau[:]
   delta_t33_m[2] = -mode_delta
   deriv_tau33 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random1,detector1,para_dw,delta_t33_p) - \
                  wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random1,detector1,para_dw,delta_t33_m))/(2* mode_delta)
   
   #tau 44 模式的微分
   delta_t44_p = delta_tau[:]
   delta_t44_p[3] = mode_delta
   delta_t44_m = delta_tau[:]
   delta_t44_m[3] = -mode_delta
   deriv_tau44 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random1,detector1,para_dw,delta_t44_p) - \
                  wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random1,detector1,para_dw,delta_t44_m))/(2* mode_delta)



   diff_vec1 = [deriv_w22,deriv_w21,deriv_w33,deriv_w44,deriv_tau22,deriv_tau21,deriv_tau33,deriv_tau44]  # Concatenate derivatives
   N_sig = 1  # Number of signals
   delta_freqency = freq_rd_gap[1]-freq_rd_gap[0]
   N_params = len(diff_vec1)  # Number of parameters we care about
   K = N_sig * N_params  # Dimension of Fisher Matrix
   fish_mix1 = np.eye(K)
   if detector1 == 'ls':
      for i in range(0,K):
         for j in range(0,K):
            fish_mix1[i,j] = dp.inner_prod(diff_vec1[i],diff_vec1[j],wf.PSD_ls(freq_rd_gap),delta_freqency)  # Construct Fisher Matrix
   elif detector1 == 'tq':
      for i in range(0,K):
         for j in range(0,K):
            fish_mix1[i,j] = dp.inner_prod(diff_vec1[i],diff_vec1[j],wf.PSD_tq(freq_rd_gap),delta_freqency)  # Construct Fisher Matrix
   else:
      for i in range(0,K):
         for j in range(0,K):
            fish_mix1[i,j] = dp.inner_prod(diff_vec1[i],diff_vec1[j],wf.PSD_tj(freq_rd_gap),delta_freqency)  # Construct Fisher Matrix


   delta_w22_p = delta_w[:]
   delta_w22_p[0] = mode_delta
   delta_w22_m = delta_w[:]
   delta_w22_m[0] = -mode_delta
   deriv_w22 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random2,detector2,delta_w22_p,para_dtau)-\
                wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random2,detector2,delta_w22_m,para_dtau))/(2* mode_delta)


   #omega 21 模式的微分
   delta_w21_p = delta_w[:]
   delta_w21_p[1] = mode_delta
   delta_w21_m = delta_w[:]
   delta_w21_m[1] = -mode_delta
   deriv_w21 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random2,detector2,delta_w21_p,para_dtau)-\
                wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random2,detector2,delta_w21_m,para_dtau))/(2* mode_delta)

   #omega 33 模式的微分
   delta_w33_p = delta_w[:]
   delta_w33_p[2] = mode_delta
   delta_w33_m = delta_w[:]
   delta_w33_m[2] = -mode_delta
   deriv_w33 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random2,detector2,delta_w33_p,para_dtau)-\
                wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random2,detector2,delta_w33_m,para_dtau))/(2* mode_delta)

   #omega 44 模式的微分
   delta_w44_p = delta_w[:]
   delta_w44_p[3] = mode_delta
   delta_w44_m = delta_w[:]
   delta_w44_m[3] = -mode_delta
   deriv_w44 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random2,detector2,delta_w44_p,para_dtau)-\
                wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random2,detector2,delta_w44_m,para_dtau))/(2* mode_delta)

   #tau 22 模式的微分
   delta_t22_p = delta_tau[:]
   delta_t22_p[0] = mode_delta
   delta_t22_m = delta_tau[:]
   delta_t22_m[0] = -mode_delta
   deriv_tau22 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random2,detector2,para_dw,delta_t22_p) - \
                  wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random2,detector2,para_dw,delta_t22_m))/(2* mode_delta)


   #tau 21 模式的微分
   delta_t21_p = delta_tau[:]
   delta_t21_p[1] = mode_delta
   delta_t21_m = delta_tau[:]
   delta_t21_m[1] = -mode_delta
   deriv_tau21 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random2,detector2,para_dw,delta_t21_p) - \
                  wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random2,detector2,para_dw,delta_t21_m))/(2* mode_delta)

   #tau 33 模式的微分
   delta_t33_p = delta_tau[:]
   delta_t33_p[2] = mode_delta
   delta_t33_m = delta_tau[:]
   delta_t33_m[2] = -mode_delta
   deriv_tau33 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random2,detector2,para_dw,delta_t33_p) - \
                  wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random2,detector2,para_dw,delta_t33_m))/(2* mode_delta)
   
   #tau 44 模式的微分
   delta_t44_p = delta_tau[:]
   delta_t44_p[3] = mode_delta
   delta_t44_m = delta_tau[:]
   delta_t44_m[3] = -mode_delta
   deriv_tau44 = (wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random2,detector2,para_dw,delta_t44_p) - \
                  wf.gap_sfa_fm(freq,paramater,angles,n_s,n_e,random2,detector2,para_dw,delta_t44_m))/(2* mode_delta)



   diff_vec2 = [deriv_w22,deriv_w21,deriv_w33,deriv_w44,deriv_tau22,deriv_tau21,deriv_tau33,deriv_tau44]  # Concatenate derivatives
   N_sig = 1  # Number of signals
   N_params = len(diff_vec2)  # Number of parameters we care about
   K = N_sig * N_params  # Dimension of Fisher Matrix
   fish_mix2 = np.eye(K)
   if detector2 == 'ls':
      for i in range(0,K):
         for j in range(0,K):
            fish_mix2[i,j] = dp.inner_prod(diff_vec2[i],diff_vec2[j],wf.PSD_ls(freq_rd_gap),delta_freqency)  # Construct Fisher Matrix
   elif detector2 == 'tj':
      for i in range(0,K):
         for j in range(0,K):
            fish_mix2[i,j] = dp.inner_prod(diff_vec2[i],diff_vec2[j],wf.PSD_tj(freq_rd_gap),delta_freqency)  # Construct Fisher Matrix
   else:
      for i in range(0,K):
         for j in range(0,K):
            fish_mix2[i,j] = dp.inner_prod(diff_vec2[i],diff_vec2[j],wf.PSD_tq(freq_rd_gap),delta_freqency)  # Construct Fisher Matrix

  # Import arbitrary precision matrix
       # mp.dps表示精确度达到的位数 
   fish_mix_prec = mp.matrix(fish_mix1) + mp.matrix(fish_mix2)
   fish_mix_inv = fish_mix_prec**-1

   #将这三行向量合成为一个矩阵
   Cov_Matrix = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         Cov_Matrix[i,j] = float(fish_mix_inv[i,j])

   #这里单独算一个第一个探测器的有数据间隙情况的fisher矩阵
   fish_mix_prec_one = mp.matrix(fish_mix1)
   fish_mix_inv_one = fish_mix_prec_one**-1

   #将这三行向量合成为一个矩阵
   Cov_Matrix_one = np.eye(K)
   for i in range(0,K):
      for j in range(0,K):
         Cov_Matrix_one[i,j] = float(fish_mix_inv_one[i,j])
         
   return np.sqrt(np.diag(Cov_Matrix))[0],np.sqrt(np.diag(Cov_Matrix))[1],np.sqrt(np.diag(Cov_Matrix))[2],np.sqrt(np.diag(Cov_Matrix))[3],\
   np.sqrt(np.diag(Cov_Matrix))[4],np.sqrt(np.diag(Cov_Matrix))[5],np.sqrt(np.diag(Cov_Matrix))[6],np.sqrt(np.diag(Cov_Matrix))[7],\
   np.sqrt(np.diag(Cov_Matrix_one))