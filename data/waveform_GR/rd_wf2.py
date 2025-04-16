#lisa
import json
import sys
import numpy
from numpy import log, exp, pi,sqrt,cos,sin,abs,log10,abs,e
import scipy.stats, scipy
import pymultinest
import matplotlib.pyplot as plt
#datafile = sys.argv[1]
#data = numpy.loadtxt('htdihltotalls.txt', dtype='complex',)
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

def  w1(m,b,a):
  return ((0.3736*(1 - 0.03135*b**2 - 0.09674*b**3 + 0.2375*b**4) + 
   2*a*(0.0629 - 0.0156*b**2 - 0.00758*b**3 - 0.0644*b**4 + 0.268*b**5 - 
      0.603*b**6))*c**3)/(cG*((10**m)*Ms))
#print(w1(1.99*10**36,0.1,0.01))  
def  tau1(m,b,a):
  return 1/((((0.0888)*(1 + 0.04371*b**2 + 0.1794*b**3 - 0.2947*b**4) + 
      2*a*(0.00099 - 0.0011*b**2 + 0.01864*b**3 - 0.17271*b**4 + 
         0.56422*b**5 - 0.8119*b**6))*c**3)/(cG*((10**m)*Ms)))
#print(tau1(1.99*10**36,0.1,0.01)) 
def  w3(m,b,a):
     return ((0.5994*(1 - 0.09911*b**2 - 0.04907*b**3 + 0.09286*b**4) + 
   3*a*(0.0674 - 0.0291*b**2 + 0.0251*b**3 - 0.3209*b**4 + 1.1703*b**5 - 
      1.3341*b**6))*c**3)/(cG*((10**m)*Ms))
#print(w3(1.99*10**36,0.1,0.01))
def  tau3(m,b,a):
  return 1/((((0.0927)*(1 + 0.07710*b**2 + 0.1399*b**3 - 0.3450*b**4) + 
     3*a*(0.00065 + 0.00023*b**2 + 0.0233*b**3 - 0.2832*b**4 + 
        1.323*b**5 - 2.442*b**6))*c**3)/(cG*((10**m)*Ms)))
#print(tau3(1.99*10**36,0.1,0.01))
Psi=pi/3
Phi0=0
Theta=0
t0=0
'''fa=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, \
0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, \
0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, \
0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, \
0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, \
0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, \
0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, \
0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, \
0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]'''
# fa=numpy.loadtxt('f.txt')
# f=numpy.array(fa)
#print(f)

'''def model1(M, v, x, t0,Phi,a,b,r):
  h=cG*M*sqrt(5/pi)*A1(v)*cos(x)*cos(Theta)*cos(2*Psi)*sin(2*Phi0)\
   *(((-cos(f*t0 + 2*Phi)*1j + sin(f*t0 + 2*Phi))*1j*tau1(M,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(M,b,a)-tau1(M,b,a)*w1(M,b,a)))\
   -((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*tau1(M,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(M,b,a) + tau1(M,b,a)*w1(M,b,a))))/(2*c**2*r)\
   +cG*M*sqrt(5/pi)*A1(v)*cos(x)*cos(2*Phi0)*sin(2*Psi)\
   *(((-cos(f*t0 + 2*Phi)*1j + sin(f*t0 + 2*Phi))*1j*tau1(M,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(M,b,a)-tau1(M,b,a)*w1(M,b,a)))\
   -((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*tau1(M,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(M,b,a) + tau1(M,b,a)*w1(M,b,a))))/(4*c**2*r)\
   +cG*M*sqrt(5/pi)*A1(v)*cos(x)*cos(Theta)**2*cos(2*Phi0)*sin(2*Psi)\
   *(((-cos(f*t0 + 2*Phi)*1j + sin(f*t0 + 2*Phi))*1j*tau1(M,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(M,b,a)-tau1(M,b,a)*w1(M,b,a)))\
   -((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*tau1(M,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(M,b,a) + tau1(M,b,a)*w1(M,b,a))))/(4*c**2*r)\
   +cG*M*sqrt(5/pi)*A1(v)*cos(2*Phi0)*cos(2*Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau1(M,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(M,b,a)-tau1(M,b,a)*w1(M,b,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau1(M,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(M,b,a) + tau1(M,b,a)*w1(M,b,a))))/(8*c**2*r)\
   +cG*M*sqrt(5/pi)*A1(v)*cos(x)**2*cos(2*Phi0)*cos(2*Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau1(M,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(M,b,a)-tau1(M,b,a)*w1(M,b,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau1(M,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(M,b,a) + tau1(M,b,a)*w1(M,b,a))))/(8*c**2*r)\
   +cG*M*sqrt(5/pi)*A1(v)*cos(Theta)**2*cos(2*Phi0)*cos(2*Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau1(M,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(M,b,a)-tau1(M,b,a)*w1(M,b,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau1(M,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(M,b,a) + tau1(M,b,a)*w1(M,b,a))))/(8*c**2*r)\
   +cG*M*sqrt(5/pi)*A1(v)*cos(x)**2*cos(Theta)**2*cos(2*Phi0)*cos(2*Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau1(M,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(M,b,a)-tau1(M,b,a)*w1(M,b,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau1(M,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(M,b,a) + tau1(M,b,a)*w1(M,b,a))))/(8*c**2*r)\
   -cG*M*sqrt(5/pi)*A1(v)*cos(Theta)*cos(Phi0)*cos(Psi)*sin(Phi0)*sin(Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau1(M,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(M,b,a)-tau1(M,b,a)*w1(M,b,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau1(M,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(M,b,a) + tau1(M,b,a)*w1(M,b,a))))/(c**2*r)\
   -cG*M*sqrt(5/pi)*A1(v)*cos(x)**2*cos(Theta)*cos(Phi0)*cos(Psi)*sin(Phi0)*sin(Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau1(M,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(M,b,a)-tau1(M,b,a)*w1(M,b,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau1(M,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(M,b,a) + tau1(M,b,a)*w1(M,b,a))))/(c**2*r)\
   -cG*M*sqrt(21/(2*pi))*A3(v)*cos(x)*cos(Theta)*cos(2*Psi)*sin(x)*sin(2*Phi0)\
   *(((-cos(f*t0 + 3*Phi)*1j + sin(f*t0 + 3*Phi))*1j*tau3(M,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(M,b,a)-tau3(M,b,a)*w3(M,b,a)))\
   -((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*tau3(M,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(M,b,a) + tau3(M,b,a)*w3(M,b,a))))/(2*c**2*r)\
   -cG*M*sqrt(21/(2*pi))*A3(v)*cos(x)*cos(2*Phi0)*sin(x)*sin(2*Psi)\
   *(((-cos(f*t0 + 3*Phi)*1j + sin(f*t0 + 3*Phi))*1j*tau3(M,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(M,b,a)-tau3(M,b,a)*w3(M,b,a)))\
   -((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*tau3(M,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(M,b,a) + tau3(M,b,a)*w3(M,b,a))))/(4*c**2*r)\
   -cG*M*sqrt(21/(2*pi))*A3(v)*cos(x)*cos(Theta)**2*cos(2*Phi0)*sin(x)*sin(2*Psi)\
   *(((-cos(f*t0 + 3*Phi)*1j + sin(f*t0 + 3*Phi))*1j*tau3(M,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(M,b,a)-tau3(M,b,a)*w3(M,b,a)))\
   -((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*tau3(M,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(M,b,a) + tau3(M,b,a)*w3(M,b,a))))/(4*c**2*r)\
   -cG*M*sqrt(21/(2*pi))*A3(v)*cos(2*Phi0)*cos(2*Psi)*sin(x)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau3(M,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(M,b,a)-tau3(M,b,a)*w3(M,b,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau3(M,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(M,b,a) + tau3(M,b,a)*w3(M,b,a))))/(8*c**2*r)\
   -cG*M*sqrt(21/(2*pi))*A3(v)*cos(x)**2*cos(2*Phi0)*cos(2*Psi)*sin(x)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau3(M,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(M,b,a)-tau3(M,b,a)*w3(M,b,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau3(M,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(M,b,a) + tau3(M,b,a)*w3(M,b,a))))/(8*c**2*r)\
   -cG*M*sqrt(21/(2*pi))*A3(v)*cos(Theta)**2*cos(2*Phi0)*cos(2*Psi)*sin(x)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau3(M,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(M,b,a)-tau3(M,b,a)*w3(M,b,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau3(M,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(M,b,a) + tau3(M,b,a)*w3(M,b,a))))/(8*c**2*r)\
   -cG*M*sqrt(21/(2*pi))*A3(v)*cos(x)**2*cos(Theta)**2*cos(2*Phi0)*cos(2*Psi)*sin(x)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau3(M,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(M,b,a)-tau3(M,b,a)*w3(M,b,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau3(M,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(M,b,a) + tau3(M,b,a)*w3(M,b,a))))/(8*c**2*r)\
   +cG*M*sqrt(21/(2*pi))*A3(v)*cos(Theta)*cos(Phi0)*cos(Psi)*sin(x)*sin(Phi0)*sin(Psi)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau3(M,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(M,b,a)-tau3(M,b,a)*w3(M,b,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau3(M,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(M,b,a) + tau3(M,b,a)*w3(M,b,a))))/(c**2*r)\
   +cG*M*sqrt(21/(2*pi))*A3(v)*cos(x)**2*cos(Theta)*cos(Phi0)*cos(Psi)*sin(x)*sin(Phi0)*sin(Psi)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau3(M,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(M,b,a)-tau3(M,b,a)*w3(M,b,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau3(M,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(M,b,a) + tau3(M,b,a)*w3(M,b,a))))/(c**2*r)
  return h'''
# model(m, a,R,v,t0,Phi,x,b):
def model(m, a,R,v,Phi,x,b,f):
  h=cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(x)*cos(Theta)*cos(2*Psi)*sin(2*Phi0)\
   *(((-cos(f*t0 + 2*Phi)*1j + sin(f*t0 + 2*Phi))*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(m,b,a)-tau1(m,b,a)*w1(m,b,a)))\
   -((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*tau1(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(m,b,a) + tau1(m,b,a)*w1(m,b,a))))/(2*c**2*R*10**9*pc)\
   +cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(x)*cos(2*Phi0)*sin(2*Psi)\
   *(((-cos(f*t0 + 2*Phi)*1j + sin(f*t0 + 2*Phi))*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(m,b,a)-tau1(m,b,a)*w1(m,b,a)))\
   -((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*tau1(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(m,b,a) + tau1(m,b,a)*w1(m,b,a))))/(4*c**2*R*10**9*pc)\
     +cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(x)*cos(Theta)**2*cos(2*Phi0)*sin(2*Psi)\
   *(((-cos(f*t0 + 2*Phi)*1j + sin(f*t0 + 2*Phi))*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(m,b,a)-tau1(m,b,a)*w1(m,b,a)))\
   -((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*tau1(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(m,b,a) + tau1(m,b,a)*w1(m,b,a))))/(4*c**2*R*10**9*pc)\
   +cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(2*Phi0)*cos(2*Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(m,b,a)-tau1(m,b,a)*w1(m,b,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(m,b,a) + tau1(m,b,a)*w1(m,b,a))))/(8*c**2*R*10**9*pc)\
   +cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(x)**2*cos(2*Phi0)*cos(2*Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(m,b,a)-tau1(m,b,a)*w1(m,b,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(m,b,a) + tau1(m,b,a)*w1(m,b,a))))/(8*c**2*R*10**9*pc)\
   +cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(Theta)**2*cos(2*Phi0)*cos(2*Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(m,b,a)-tau1(m,b,a)*w1(m,b,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(m,b,a) + tau1(m,b,a)*w1(m,b,a))))/(8*c**2*R*10**9*pc)\
   +cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(x)**2*cos(Theta)**2*cos(2*Phi0)*cos(2*Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(m,b,a)-tau1(m,b,a)*w1(m,b,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(m,b,a) + tau1(m,b,a)*w1(m,b,a))))/(8*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(Theta)*cos(Phi0)*cos(Psi)*sin(Phi0)*sin(Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(m,b,a)-tau1(m,b,a)*w1(m,b,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(m,b,a) + tau1(m,b,a)*w1(m,b,a))))/(c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(x)**2*cos(Theta)*cos(Phi0)*cos(Psi)*sin(Phi0)*sin(Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(m,b,a)-tau1(m,b,a)*w1(m,b,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(m,b,a) + tau1(m,b,a)*w1(m,b,a))))/(c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(x)*cos(Theta)*cos(2*Psi)*sin(x)*sin(2*Phi0)\
   *(((-cos(f*t0 + 3*Phi)*1j + sin(f*t0 + 3*Phi))*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(m,b,a)-tau3(m,b,a)*w3(m,b,a)))\
   -((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*tau3(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(m,b,a) + tau3(m,b,a)*w3(m,b,a))))/(2*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(x)*cos(2*Phi0)*sin(x)*sin(2*Psi)\
   *(((-cos(f*t0 + 3*Phi)*1j + sin(f*t0 + 3*Phi))*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(m,b,a)-tau3(m,b,a)*w3(m,b,a)))\
   -((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*tau3(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(m,b,a) + tau3(m,b,a)*w3(m,b,a))))/(4*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(x)*cos(Theta)**2*cos(2*Phi0)*sin(x)*sin(2*Psi)\
   *(((-cos(f*t0 + 3*Phi)*1j + sin(f*t0 + 3*Phi))*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(m,b,a)-tau3(m,b,a)*w3(m,b,a)))\
   -((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*tau3(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(m,b,a) + tau3(m,b,a)*w3(m,b,a))))/(4*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(2*Phi0)*cos(2*Psi)*sin(x)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(m,b,a)-tau3(m,b,a)*w3(m,b,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(m,b,a) + tau3(m,b,a)*w3(m,b,a))))/(8*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(x)**2*cos(2*Phi0)*cos(2*Psi)*sin(x)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(m,b,a)-tau3(m,b,a)*w3(m,b,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(m,b,a) + tau3(m,b,a)*w3(m,b,a))))/(8*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(Theta)**2*cos(2*Phi0)*cos(2*Psi)*sin(x)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(m,b,a)-tau3(m,b,a)*w3(m,b,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(m,b,a) + tau3(m,b,a)*w3(m,b,a))))/(8*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(x)**2*cos(Theta)**2*cos(2*Phi0)*cos(2*Psi)*sin(x)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(m,b,a)-tau3(m,b,a)*w3(m,b,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(m,b,a) + tau3(m,b,a)*w3(m,b,a))))/(8*c**2*R*10**9*pc)\
   +cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(Theta)*cos(Phi0)*cos(Psi)*sin(x)*sin(Phi0)*sin(Psi)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(m,b,a)-tau3(m,b,a)*w3(m,b,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(m,b,a) + tau3(m,b,a)*w3(m,b,a))))/(c**2*R*10**9*pc)\
   +cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(x)**2*cos(Theta)*cos(Phi0)*cos(Psi)*sin(x)*sin(Phi0)*sin(Psi)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(m,b,a)-tau3(m,b,a)*w3(m,b,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(m,b,a) + tau3(m,b,a)*w3(m,b,a))))/(c**2*R*10**9*pc)

  return h
   
   
'''   +cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(x)*cos(Theta)**2*cos(2*Phi0)*sin(2*Psi)\
   *(((-cos(f*t0 + 2*Phi)*1j + sin(f*t0 + 2*Phi))*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(m,b,a)-tau1(m,b,a)*w1(m,b,a)))\
   -((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*tau1(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(m,b,a) + tau1(m,b,a)*w1(m,b,a))))/(4*c**2*R*10**9*pc)\
   +cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(2*Phi0)*cos(2*Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(m,b,a)-tau1(m,b,a)*w1(m,b,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(m,b,a) + tau1(m,b,a)*w1(m,b,a))))/(8*c**2*R*10**9*pc)\
   +cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(x)**2*cos(2*Phi0)*cos(2*Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(m,b,a)-tau1(m,b,a)*w1(m,b,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(m,b,a) + tau1(m,b,a)*w1(m,b,a))))/(8*c**2*R*10**9*pc)\
   +cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(Theta)**2*cos(2*Phi0)*cos(2*Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(m,b,a)-tau1(m,b,a)*w1(m,b,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(m,b,a) + tau1(m,b,a)*w1(m,b,a))))/(8*c**2*R*10**9*pc)\
   +cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(x)**2*cos(Theta)**2*cos(2*Phi0)*cos(2*Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(m,b,a)-tau1(m,b,a)*w1(m,b,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(m,b,a) + tau1(m,b,a)*w1(m,b,a))))/(8*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(Theta)*cos(Phi0)*cos(Psi)*sin(Phi0)*sin(Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(m,b,a)-tau1(m,b,a)*w1(m,b,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(m,b,a) + tau1(m,b,a)*w1(m,b,a))))/(c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(5/pi)*A1(v)*cos(x)**2*cos(Theta)*cos(Phi0)*cos(Psi)*sin(Phi0)*sin(Psi)\
   *(((cos(f*t0 + 2*Phi) + sin(f*t0 + 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau1(m,b,a)-tau1(m,b,a)*w1(m,b,a)))\
   +((cos(f*t0 - 2*Phi) + sin(f*t0 - 2*Phi)*1j)*1j*tau1(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau1(m,b,a) + tau1(m,b,a)*w1(m,b,a))))/(c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(x)*cos(Theta)*cos(2*Psi)*sin(x)*sin(2*Phi0)\
   *(((-cos(f*t0 + 3*Phi)*1j + sin(f*t0 + 3*Phi))*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(m,b,a)-tau3(m,b,a)*w3(m,b,a)))\
   -((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*tau3(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(m,b,a) + tau3(m,b,a)*w3(m,b,a))))/(2*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(x)*cos(2*Phi0)*sin(x)*sin(2*Psi)\
   *(((-cos(f*t0 + 3*Phi)*1j + sin(f*t0 + 3*Phi))*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(m,b,a)-tau3(m,b,a)*w3(m,b,a)))\
   -((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*tau3(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(m,b,a) + tau3(m,b,a)*w3(m,b,a))))/(4*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(x)*cos(Theta)**2*cos(2*Phi0)*sin(x)*sin(2*Psi)\
   *(((-cos(f*t0 + 3*Phi)*1j + sin(f*t0 + 3*Phi))*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(m,b,a)-tau3(m,b,a)*w3(m,b,a)))\
   -((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*tau3(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(m,b,a) + tau3(m,b,a)*w3(m,b,a))))/(4*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(2*Phi0)*cos(2*Psi)*sin(x)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(m,b,a)-tau3(m,b,a)*w3(m,b,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(m,b,a) + tau3(m,b,a)*w3(m,b,a))))/(8*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(x)**2*cos(2*Phi0)*cos(2*Psi)*sin(x)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(m,b,a)-tau3(m,b,a)*w3(m,b,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(m,b,a) + tau3(m,b,a)*w3(m,b,a))))/(8*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(Theta)**2*cos(2*Phi0)*cos(2*Psi)*sin(x)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(m,b,a)-tau3(m,b,a)*w3(m,b,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(m,b,a) + tau3(m,b,a)*w3(m,b,a))))/(8*c**2*R*10**9*pc)\
   -cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(x)**2*cos(Theta)**2*cos(2*Phi0)*cos(2*Psi)*sin(x)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(m,b,a)-tau3(m,b,a)*w3(m,b,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(m,b,a) + tau3(m,b,a)*w3(m,b,a))))/(8*c**2*R*10**9*pc)\
   +cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(Theta)*cos(Phi0)*cos(Psi)*sin(x)*sin(Phi0)*sin(Psi)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(m,b,a)-tau3(m,b,a)*w3(m,b,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(m,b,a) + tau3(m,b,a)*w3(m,b,a))))/(c**2*R*10**9*pc)\
   +cG*(10**m)*Ms*sqrt(21/(2*pi))*A3(v)*cos(x)**2*cos(Theta)*cos(Phi0)*cos(Psi)*sin(x)*sin(Phi0)*sin(Psi)\
   *(((cos(f*t0 + 3*Phi) + sin(f*t0 + 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j+f*tau3(m,b,a)-tau3(m,b,a)*w3(m,b,a)))\
   +((cos(f*t0 - 3*Phi) + sin(f*t0 - 3*Phi)*1j)*1j*tau3(m,b,a))/(2*sqrt(2*pi)*(1j + f*tau3(m,b,a) + tau3(m,b,a)*w3(m,b,a))))/(c**2*R*10**9*pc)'''
#print(model(10**6*Ms, 2/9, pi/3, 0,0,0.01,0.1,2.5*10**9*pc))
'''noisea=[6.026930881180586*10**-20, 6.001203405331356*10**-20, \
6.000190141567148*10**-20, 6.000075197347526*10**-20, \
6.000043119730023*10**-20, 6.0000237648569184*10**-20, \
6.000011224097523*10**-20, 6.0000046995093765*10**-20, \
6.000002347092599*10**-20, 6.000001924902162*10**-20, \
6.000001840622609*10**-20, 6.000001485256807*10**-20, \
6.000000943540367*10**-20, 6.000000501063959*10**-20, \
6.000000304179407*10**-20, 6.000000293713936*10**-20, \
6.0000003226536065*10**-20, 6.000000293382278*10**-20, \
6.0000002067841895*10**-20, 6.000000120304959*10**-20, \
6.000000079180121*10**-20, 6.000000082169849*10**-20, \
6.000000096298557*10**-20, 6.000000092827787*10**-20, \
6.000000068987577*10**-20, 6.0000000421220415*10**-20, \
6.000000028975948*10**-20, 6.000000031316307*10**-20, \
6.000000038101199*10**-20, 6.000000038022223*10**-20, \
6.0000000291798976*10**-20, 6.000000018357035*10**-20, \
6.000000012984841*10**-20, 6.000000014404119*10**-20, \
6.000000017958009*10**-20, 6.0000000183363244*10**-20, \
6.000000014378826*10**-20, 6.000000009231406*10**-20, \
6.000000006656314*10**-20, 6.000000007519037*10**-20, \
6.000000009536627*10**-20, 6.000000009897488*10**-20, \
6.000000007882362*10**-20, 6.0000000051356026*10**-20, \
6.000000003755277*10**-20, 6.000000004299033*10**-20, \
6.000000005522537*10**-20, 6.0000000058017246*10**-20, \
6.000000004674614*10**-20, 6.0000000030797965*10**-20, \
6.0000000022762045*10**-20, 6.000000002632623*10**-20, \
6.000000003415282*10**-20, 6.000000003621987*10**-20, \
6.000000002944959*10**-20, 6.000000001957266*10**-20, \
6.000000001458789*10**-20, 6.0000000017009435*10**-20, \
6.0000000022239336*10**-20, 6.000000002376385*10**-20, \
6.000000001946303*10**-20, 6.0000000013026725*10**-20, \
6.000000000977529*10**-20, 6.000000001147313*10**-20, \
6.0000000015096505*10**-20, 6.000000001623103*10**-20, \
6.0000000013373054*10**-20, 6.000000000900256*10**-20, \
6.000000000679353*10**-20, 6.000000000801696*10**-20, \
6.000000001060465*10**-20, 6.000000001146019*10**-20, \
6.000000000948939*10**-20, 6.000000000641911*10**-20, \
6.000000000486683*10**-20, 6.000000000576962*10**-20, \
6.000000000766597*10**-20, 6.0000000008320384*10**-20, \
6.000000000691865*10**-20, 6.0000000004699395*10**-20, \
6.000000000357726*10**-20, 6.000000000425742*10**-20, \
6.000000000567829*10**-20, 6.000000000618592*10**-20, \
6.000000000516242*10**-20, 6.0000000003518905*10**-20, \
6.0000000002687904*10**-20, 6.000000000320975*10**-20, \
6.000000000429506*10**-20, 6.000000000469409*10**-20, \
6.0000000003929745*10**-20, 6.0000000002686896*10**-20, \
6.000000000205854*10**-20, 6.0000000002465414*10**-20, \
6.000000000330853*10**-20, 6.000000000362607*10**-20, \
6.000000000304398*10**-20, 6.000000000208688*10**-20, \
6.0000000001603064*10**-20, 6.000000000192487*10**-20]'''
# noisea = numpy.loadtxt('htdihlnoils.txt')
# noise=numpy.array(noisea)
# L=sqrt(3)*10**8#tq:sqrt(3)*10**8#tj:3*10**9#2*sqrt(3)*0.0048*1.496*10**11
# sa=1*10**(-15)#tq:1*10**(-15)#3*10**(-15)
# sx=1*10**(-12)#tq:1*10**(-12)#tj:8*10**(-12)#1.5*10**(-11)
# u=2*pi*f*L/c
# factor = 10**5
# noise = sqrt(factor*sx**2/L**2+4*sa**2*(1+10**-4/f) /((2*pi*f)**4*L**2))
# #sqrt(factor*4*sin(u)**2/L**2 *(sx**2 + sa**2 /(2*pi*f)**4*(3+cos(2*u))) )
# hmodel = model(6.5,0.1,0.5, 2/9, 0, pi/3, 0.2)
# data = hmodel + numpy.random.normal(loc=numpy.zeros_like(f), scale=noise, size=None)
# #(m, a,R,v,t0,Phi,x,b)
# #m, a,R,v,Phi,x,b
# #v = 2/9; Phi0 = 0; Psi = pi/3;Phi= 0;Theta= 0; r = 2.5*10^9 pc;a=0.01;t0= 0;x=pi]/3;b = 0.1;M = 10^6 Ms;
# def prior(cube, ndim, nparams):
#     cube[0] = cube[0]*1+6 # uniform prior between M = 10^5.698970004336018` Ms; 1 and 10^11
#     cube[1] = cube[1]*0.2+0.001  #  a=0.01 0.001 and 0.051 #  v = 2/9; 1/9 and 3/9
#     cube[2] = cube[1]*2+0.2     # tq:4.5+9,*2+ 0.2 
#     cube[3] = cube[3]*0.25    #v = 2/9
#     #cube[4] = cube[3]*0.1   #  t0= 0 
#     cube[4] = cube[4]*2*pi #  Phi= 0
#     cube[5] = cube[5]*pi # x=pi
#     cube[6] = cube[6]*0.4  #b = 0.1
# '''def prior(cube, ndim, nparams):
#     cube[0] = 10**(cube[0]*7) # uniform prior between M = 10^6 Ms; 1 and 10^7
# 	  cube[1] = cube[1]*2/9+1/9 #  v = 2/9; 1/9 and 3/9
# 	  cube[2] = cube[1]*pi
# 	  cube[3] = cube[3]*0.01
#     cube[4] = cube[4]*2*pi    #  Phi= 0 0 and 2pi
#     cube[5] = cube[5]*0.05+0.001   #  a=0.01 0.001 and 0.051
#     cube[6] = cube[6]*0.1+0.05 # b = 0.1 0.05 and 0.15
#     cube[7] = ((cube[7]+2)*10**9*pc)  #uniform prior between r = 2.5*10^9 pc  2*10^9 pc and 3*10^9 pc'''
# #m, a,R,v,t0,Phi,x,b
# #model(m, a,R,v,Phi,x,b)
# def loglike(cube, ndim, nparams):
#   m,a,R,v,Phi,x,b = cube[0], cube[1], cube[2],cube[3],cube[4],cube[5],cube[6]#,cube[7]
#   ymodel = model(m,a,R,v,Phi,x,b)
#   # loglikelihood = exp ((-0.5 * abs((ymodel - data) / (10**3*noise))**2 - (-0.5 * abs((model(6.5, 0.1, 5, 2/9, 0, pi/3, 0.1) - data) / (10**3*noise))**2) ) .sum() )
#   # loglikelihood =  -log( (0.5* abs(ymodel - data) **2/(noise)**2 ).sum() )
#   loglikelihood =  -0.5* ( abs(ymodel - data) **2/(noise)**2 ).sum() 
#   return loglikelihood

# # number of dimensions our problem has
# #"$Log({M_Z}/{M_ \odot })$", "${\chi _f}$","$Log({D_L}/Gpc)$","$\nu $","${{t_0}}$","$\phi $","$\iota $","$\zeta $"
# parameters = ["$Lo{g_{10}}({M_Z}/{M_ \odot })$", "${\chi _f}$","${D_L}/Gpc$","$v$","$\phi $","$\iota $","$\zeta $"]
# n_params = len(parameters)
