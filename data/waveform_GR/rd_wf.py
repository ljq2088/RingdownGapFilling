Ms = 1.99*10**30
# Re-run everything due to code execution state reset

import numpy as np
Ms = 1.99*10**30
# 定义模态参数矩阵
# 每行：[f1, f2, f3, q1, q2, q3]，对应 (2,1), (2,2), (3,3), (4,4)
mode_matrix = np.array([
    [0.6000, -0.2339, 0.4175, -0.3000, 2.3561, -0.2277],
    [1.5251, -1.1568, 0.1292,  0.7000, 1.4187, -0.4990],
    [1.8956, -1.3043, 0.1818,  0.9000, 2.3430, -0.4810],
    [2.3000, -1.5056, 0.2244,  1.1929, 3.1191, -0.4825],
])

# 对应模态标签
modes = [(2, 1), (2, 2), (3, 3), (4, 4)]
pc = 3.26*9.46*10**15
Ms = 1.99*10**30
c = 3*10**8
cG = 6.67*10**-11
# 封装返回关于 M, chi 的 QNM 函数
def get_qnm_functions(mode_matrix, modes):
    """
    为每个模式生成一个返回 ω_GR 和 τ_GR 的函数，保存在字典中
    """
    qnm_functions = {}

    for i, (l, m) in enumerate(modes):
        f1, f2, f3, q1, q2, q3 = mode_matrix[i]

        def qnm_func(M, chi, f1=f1, f2=f2, f3=f3, q1=q1, q2=q2, q3=q3):
            omega = (f1 + f2 * (1 - chi)**f3) / ((10**M)*Ms)/cG*c**3
            tau = 2 * (q1 + q2 * (1 - chi)**q3) / omega
            return omega, tau

        qnm_functions[(l, m)] = qnm_func

    return qnm_functions

# 生成 QNM 函数字典
qnm_dict = get_qnm_functions(mode_matrix, modes)

# 示例：计算 (3,3) 模式在 M=100, chi=0.8 下的 QNM 频率与衰减时间
#omega_33, tau_33 = qnm_dict[(3, 3)](M=6, chi=0.8)
def w22(m,a):
    return qnm_dict[(2, 2)](m,a)[0]
def w33(m,a):
    return qnm_dict[(3, 3)](m,a)[0]
def tau22(m,a):
    return qnm_dict[(2, 2)](m,a)[1]
def tau33(m,a):
    return qnm_dict[(3, 3)](m,a)[1]
#lisa
import json
import sys
import numpy
from numpy import log, exp, pi,sqrt,cos,sin,abs,log10,abs,e
import scipy.stats, scipy
import pymultinest
import matplotlib.pyplot as plt
#datafile = sys.argv[1]

#np.loadtxt('./data/GammaT_mean.txt', dtype='complex')
#data = numpy.genfromtxt('htdihltotalls.txt',delimiter=',',dtype='str')
#mapping = numpy.vectorize(lambda t:complex())
#p1= mapping(data)
#beam_data = pd.read_csv("beam1.csv").astype('complex')
#print(data)
'''olds = ['i', '^']
news = ['j', '*']
filename='htdihltotalls.csv'#csv数据为复数
temp = numpy.genfromtxt(filename, delimiter=',',dtype='str')
mapping = numpy.vectorize(lambda t:complex(t.replace(olds,news)))
#mapping = numpy.vectorize(lambda t:complex(t.replace('i','j'),t.replace('^','*')))
p1= mapping(temp)'''

pc = 3.26*9.46*10**15
Ms = 1.99*10**30
c = 3*10**8
cG = 6.67*10**-11
def A1(v):
    return 0.864*v

def  A3(v):
    return 0.44*(1 - 4*v)**0.45*A1(v)
#print(A3(0.1))




#print(tau3(1.99*10**36,0.1,0.01))
Psi=pi/3
Phi0=0
Theta=0
t0=0


def model(m, a,R,v,Phi,x,f):
  h=cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(x)*cos(Theta)*cos(2*Psi)*sin(2*Phi0)\
   *(((-cos(f*t0 + 2*Phi)*1j + sin(f*t0 + 2*Phi))*1j*tau22(m,a))/(2*sqrt(2*pi)*(1j+f*tau22(m,a)-tau22(m,a)*w22(m,a)))\
   -((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*tau22(m,a))/(2*sqrt(2*pi)*(1j + f*tau22(m,a) + tau22(m,a)*w22(m,a))))/(2*c**2*R*10**9*pc)\
   +cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(x)*cos(2*Phi0)*sin(2*Psi)\
   *(((-cos(f*t0 + 2*Phi)*1j + sin(f*t0 + 2*Phi))*1j*tau22(m,a))/(2*sqrt(2*pi)*(1j+f*tau22(m,a)-tau22(m,a)*w22(m,a)))\
   -((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*tau22(m,a))/(2*sqrt(2*pi)*(1j + f*tau22(m,a) + tau22(m,a)*w22(m,a))))/(4*c**2*R*10**9*pc)\
     +cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(x)*cos(Theta)**2*cos(2*Phi0)*sin(2*Psi)\
   *(((-cos(f*t0 + 2*Phi)*1j + sin(f*t0 + 2*Phi))*1j*tau22(m,a))/(2*sqrt(2*pi)*(1j+f*tau22(m,a)-tau22(m,a)*w22(m,a)))\
   -((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*tau22(m,a))/(2*sqrt(2*pi)*(1j + f*tau22(m,a) + tau22(m,a)*w22(m,a))))/(4*c**2*R*10**9*pc)\
   +cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(2*Phi0)*cos(2*Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau22(m,a))/(2*sqrt(2*pi)*(1j+f*tau22(m,a)-tau22(m,a)*w22(m,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau22(m,a))/(2*sqrt(2*pi)*(1j + f*tau22(m,a) + tau22(m,a)*w22(m,a))))/(8*c**2*R*10**9*pc)\
   +cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(x)**2*cos(2*Phi0)*cos(2*Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau22(m,a))/(2*sqrt(2*pi)*(1j+f*tau22(m,a)-tau22(m,a)*w22(m,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau22(m,a))/(2*sqrt(2*pi)*(1j + f*tau22(m,a) + tau22(m,a)*w22(m,a))))/(8*c**2*R*10**9*pc)\
   +cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(Theta)**2*cos(2*Phi0)*cos(2*Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau22(m,a))/(2*sqrt(2*pi)*(1j+f*tau22(m,a)-tau22(m,a)*w22(m,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau22(m,a))/(2*sqrt(2*pi)*(1j + f*tau22(m,a) + tau22(m,a)*w22(m,a))))/(8*c**2*R*10**9*pc)\
   +cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(x)**2*cos(Theta)**2*cos(2*Phi0)*cos(2*Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau22(m,a))/(2*sqrt(2*pi)*(1j+f*tau22(m,a)-tau22(m,a)*w22(m,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau22(m,a))/(2*sqrt(2*pi)*(1j + f*tau22(m,a) + tau22(m,a)*w22(m,a))))/(8*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(Theta)*cos(Phi0)*cos(Psi)*sin(Phi0)*sin(Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau22(m,a))/(2*sqrt(2*pi)*(1j+f*tau22(m,a)-tau22(m,a)*w22(m,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau22(m,a))/(2*sqrt(2*pi)*(1j + f*tau22(m,a) + tau22(m,a)*w22(m,a))))/(c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(x)**2*cos(Theta)*cos(Phi0)*cos(Psi)*sin(Phi0)*sin(Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau22(m,a))/(2*sqrt(2*pi)*(1j+f*tau22(m,a)-tau22(m,a)*w22(m,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau22(m,a))/(2*sqrt(2*pi)*(1j + f*tau22(m,a) + tau22(m,a)*w22(m,a))))/(c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(x)*cos(Theta)*cos(2*Psi)*sin(x)*sin(2*Phi0)\
   *(((-cos(f*t0 + 3*Phi)*1j + sin(f*t0 + 3*Phi))*1j*tau33(m,a))/(2*sqrt(2*pi)*(1j+f*tau33(m,a)-tau33(m,a)*w33(m,a)))\
   -((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*tau33(m,a))/(2*sqrt(2*pi)*(1j + f*tau33(m,a) + tau33(m,a)*w33(m,a))))/(2*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(x)*cos(2*Phi0)*sin(x)*sin(2*Psi)\
   *(((-cos(f*t0 + 3*Phi)*1j + sin(f*t0 + 3*Phi))*1j*tau33(m,a))/(2*sqrt(2*pi)*(1j+f*tau33(m,a)-tau33(m,a)*w33(m,a)))\
   -((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*tau33(m,a))/(2*sqrt(2*pi)*(1j + f*tau33(m,a) + tau33(m,a)*w33(m,a))))/(4*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(x)*cos(Theta)**2*cos(2*Phi0)*sin(x)*sin(2*Psi)\
   *(((-cos(f*t0 + 3*Phi)*1j + sin(f*t0 + 3*Phi))*1j*tau33(m,a))/(2*sqrt(2*pi)*(1j+f*tau33(m,a)-tau33(m,a)*w33(m,a)))\
   -((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*tau33(m,a))/(2*sqrt(2*pi)*(1j + f*tau33(m,a) + tau33(m,a)*w33(m,a))))/(4*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(2*Phi0)*cos(2*Psi)*sin(x)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau33(m,a))/(2*sqrt(2*pi)*(1j+f*tau33(m,a)-tau33(m,a)*w33(m,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau33(m,a))/(2*sqrt(2*pi)*(1j + f*tau33(m,a) + tau33(m,a)*w33(m,a))))/(8*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(x)**2*cos(2*Phi0)*cos(2*Psi)*sin(x)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau33(m,a))/(2*sqrt(2*pi)*(1j+f*tau33(m,a)-tau33(m,a)*w33(m,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau33(m,a))/(2*sqrt(2*pi)*(1j + f*tau33(m,a) + tau33(m,a)*w33(m,a))))/(8*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(Theta)**2*cos(2*Phi0)*cos(2*Psi)*sin(x)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau33(m,a))/(2*sqrt(2*pi)*(1j+f*tau33(m,a)-tau33(m,a)*w33(m,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau33(m,a))/(2*sqrt(2*pi)*(1j + f*tau33(m,a) + tau33(m,a)*w33(m,a))))/(8*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(x)**2*cos(Theta)**2*cos(2*Phi0)*cos(2*Psi)*sin(x)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau33(m,a))/(2*sqrt(2*pi)*(1j+f*tau33(m,a)-tau33(m,a)*w33(m,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau33(m,a))/(2*sqrt(2*pi)*(1j + f*tau33(m,a) + tau33(m,a)*w33(m,a))))/(8*c**2*R*10**9*pc)\
   +cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(Theta)*cos(Phi0)*cos(Psi)*sin(x)*sin(Phi0)*sin(Psi)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau33(m,a))/(2*sqrt(2*pi)*(1j+f*tau33(m,a)-tau33(m,a)*w33(m,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau33(m,a))/(2*sqrt(2*pi)*(1j + f*tau33(m,a) + tau33(m,a)*w33(m,a))))/(c**2*R*10**9*pc)\
   +cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(x)**2*cos(Theta)*cos(Phi0)*cos(Psi)*sin(x)*sin(Phi0)*sin(Psi)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau33(m,a))/(2*sqrt(2*pi)*(1j+f*tau33(m,a)-tau33(m,a)*w33(m,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau33(m,a))/(2*sqrt(2*pi)*(1j + f*tau33(m,a) + tau33(m,a)*w33(m,a))))/(c**2*R*10**9*pc)

  return h
   
   
