import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
# Bunch of units

GM_sun = 1.3271244*1e20
c =2.9979246*1e8
M_sun =1.9884099*1e30
G = 6.6743*1e-11
pc= 3.0856776*1e16
pi = np.pi
Mpc = (10**6) * pc
def htilde(f,eps,params):
    
    """
    Here we calculate a TaylorF2 model up to 2PN which takes as input the following
    set of parameters: (log of chirp mass, symmetric mass ratio, beta).
    This can easily be changed in the first few lines where the parameters are loaded.
    The main reference is https://arxiv.org/pdf/gr-qc/0509116.pdf [Eqs (3.4)].
    
    Note on spin: 
    
    The spin parameter beta is defined in Eq.(2.3a) in [arxiv:0411129].
    Notice that this quantity is constructed in such a way to be smaller or equal
    than 9.4, and of course it ranges from 0 (no spins) to this upper value. 
    The coefficient enters the phase as in Eq.(2.2) in the same paper.
    """
    # Units
    
    # Load the parameters
    t0 =1.
    phi0 =0.
    Mchirp_true = M_sun * np.exp(params[0])
    eta_true = params[1]
    beta_true = params[2]
    Deff = params[3]
    theta = -11831/9240 #in PN coefficients!
    delta = -1987/3080  #in PN coefficients!
    # PN expansion parameter (velocity).
    
    v = (pi*G*Mchirp_true*eta_true**(-3/5)/(c**3) * f)**(1/3)
    # Amplitude explicitly given in terms of units and frequency.
    # Notice that lowest PN order here is fine. Biggest contributions from phase.
    
    amplitude_1 = - (Mpc/Deff)*np.sqrt((5/(24*pi)))*(GM_sun/(c**2 *Mpc))
    amplitude_2 = (pi*GM_sun/(c**3))**(-1/6) * (Mchirp_true/M_sun)**(5/6)
    amplitude = amplitude_1*amplitude_2 * f**(-7/6)
    # Phase: add or remove PN orders here as you see fit.
    
    psi_const = 2*pi*f*t0 - 2*phi0 - pi/4
    psi1PN = (3715/756 + (55/9)*eta_true)*v**(-3)
    psi1_5PN_tails = -16*pi*v**(-2)
    psi1_5PN_spin = 4*beta_true*v**(-2)
    
    psi2PN = (15293365/508032+(27145/504)*eta_true+(3085/72)*eta_true**2)*v**(-1)
    psi25PNlog = pi*(38645/252- (65/3) *eta_true)* np.log(v)
    psi3PN = v*(11583231236531/4694215680 - (640/3) * (pi**2) -6848/21 *np.euler_gamma
              + eta_true*(-15335597827/3048192 + (2255/12) * (pi**2) - 1760/3 * theta - 12320/9 * delta)
              + (eta_true**2) *76055/1728 - (eta_true**3) * 127825/1296 - 6848/21 * np.log(4))
    psi3PNlog = - 6848/21 *v * np.log(v)
    psi35PN = pi * v**2 * (77096675./254016 + (378515./1512) *eta_true - 74045./756 * (eta_true**2)* (1-eps))
    psi_fullPN = (3/(128*eta_true))*(v**(-5)+psi1PN+psi1_5PN_tails+psi1_5PN_spin+psi2PN
                                  + psi25PNlog + psi3PN + psi3PNlog + psi35PN)
    psi = psi_const + psi_fullPN 
    return amplitude* np.exp(-1j*psi)

def htilde_GB(f,params):
    
    """
    Here we calculate a TaylorF2 model up to 2PN which takes as input the following
    set of parameters: (log of chirp mass, symmetric mass ratio, beta).
    This can easily be changed in the first few lines where the parameters are loaded.
    The main reference is https://arxiv.org/pdf/gr-qc/0509116.pdf [Eqs (3.4)].
    
    Note on spin: 
    
    The spin parameter beta is defined in Eq.(2.3a) in [arxiv:0411129].
    Notice that this quantity is constructed in such a way to be smaller or equal
    than 9.4, and of course it ranges from 0 (no spins) to this upper value. 
    The coefficient enters the phase as in Eq.(2.2) in the same paper.
    """
    # Units
    t0 =1.
    phi0 =0.
    
    # Load the parameters
    Mchirp_true = M_sun * np.exp(params[0])
    eta_true = params[1]
    Deff = params[2]
    

    # PN expansion parameter (velocity).
    
    v = (pi*G*Mchirp_true*eta_true**(-3/5)/(c**3) * f)**(1/3)
    # Amplitude explicitly given in terms of units and frequency.
    # Notice that lowest PN order here is fine. Biggest contributions from phase.
    
    amplitude_1 = - (1/Deff)*np.sqrt((5/(24*pi)))*(GM_sun/(c**2 ))
    amplitude_2 = (pi*GM_sun/(c**3))**(-1/6) * (Mchirp_true/M_sun)**(5/6)
    amplitude = amplitude_1*amplitude_2 * f**(-7/6)
    
    new_amplitude = -np.sqrt(5*np.pi/24)*(G*Mchirp_true/(c**3))*(G*Mchirp_true/(Deff*c**2))*(f*np.pi*G*Mchirp_true/(c**3))**(-7/6)
    
    
    psi_const = 2*pi*f*t0 - 2*phi0 - pi/4
    psi_fullPN = (3/(128*eta_true))*(v**(-5) )

    psi = psi_const + psi_fullPN 
    return amplitude_1,amplitude_2,np.exp(-1j*psi),new_amplitude* np.exp(-1j*psi)





def T_chirp(fmin,M_chirp,eta):

    # M = (m1 + m2)*M_sun
    M_chirp *= M_sun
    
    M = M_chirp*eta**(-3/5)
    v_low = (pi*G*M_chirp*eta**(-3/5)/(c**3) * fmin)**(1/3)
    
    theta = -11831/9240 #in PN coefficients!
    delta = -1987/3080  #in PN coefficients!
    gamma = np.euler_gamma
    
    pre_fact = ((5/(256*eta)) * G*M/(c**3))
    first_term = (v_low**(-8) + (743/252 + (11/3) * eta ) * (v_low **(-6)) - (32*np.pi/5)*v_low**(-5)
                +(3058673/508032 + (5429/504)*eta + (617/72)*eta**2)*v_low**(-4)
                 +(13*eta/3 - 7729/252)*np.pi*v_low**-3)
    
    second_term = (6848*gamma/105 - 10052469856691/23471078400 + 128*pi**2/3 + (
    3147553127/3048192 - 451*(pi**2)/12)*eta - (15211*eta**2)/1728 + (2555*eta**3 / 1296) +
                   (6848/105)*np.log(4*v_low))*v_low**-2
    
    third_term = ((14809/378)*eta**2 - (75703/756) * eta - 15419335/127008)*pi*v_low**-1
    return pre_fact * (first_term + second_term + third_term)

def final_frequency(M_chirp,eta):
    M_tot = M_chirp*eta**(-3/5) * M_sun
    
    return (c**3)/(6*np.sqrt(6)*np.pi*G*M_tot)

"""
Signal parameters
"""

# # Fix these two impostors, assume they are known perfectly.
# t0 =1.
# phi0 =0.
# fmin = 1e-4

# # variables to sample through

# Deff_1 = 2 * 1e3 * Mpc
# beta_1 = 6
# m1 = 1e6  
# m2 = 2*1e6
# M_tot_1 = (m1 + m2)  # M_tot in kilograms
# eta_1 = (m1*m2)/(M_tot_1**2)  # Symmetric mass ratio [dimensionless]=
# M_chirp_1 = M_tot_1*eta_1**(3/5)  # Chirp mass in units of kilograms 


# f_max_1 = final_frequency(M_chirp_1,eta_1)  # Calculate maximum frequency (Schwarzschild ISCO frequency)
# t_max_1 = T_chirp(fmin,M_chirp_1,eta_1)     # Calculate maximum chirping time that binary radiates stuff



# logMchirp_1 = np.log(M_chirp_1)






# # True waveforms have eps_GR = 0 and approximate wavefors eps_AP \neq eps_GR.
# eps_GR = 0
# eps_AP = (4*1*1e-2)

# # Calculate max frequency and chirp time

# fmax = f_max_1 # Compute "smallest" largest frequency

# tmax = t_max_1 # Compute maximum chirping time for both binaries



# delta_t = 1/(2*fmax)         # Set sampling interval so that we can resolved frequencies of BOTH signals

# t = np.arange(0,tmax,delta_t)     
# n_t = len(t)                      # Extract length

# delta_f = 1/(n_t*delta_t)         # Extract sampling frequency

# freq_bin = np.arange(fmin,fmax,delta_f)     # Extract frequency series
# n_f = len(freq_bin)                         # Extract length of frequency domain series


def h_model(Deff,m1,m2,fmin):
    """
    Parameters
    ----------
    Deff : float
        effective distance of the binary
    m1 : float
        mass of the first black hole
    m2 : float
        mass of the second black hole
    fmin : float
        initial frequency of the ringdown signal

    Returns
    -------
    freq_bin : array_like
        frequency series
    h : array_like
        strain signal
    """
    Deff= Deff * Mpc*1e3  # Convert Deff to meters
    M_tot = (m1 + m2)  # M_tot in kilograms
    eta = (m1*m2)/(M_tot**2)  # Symmetric mass ratio [dimensionless]=
    beta= 6
    M_chirp = M_tot*eta**(3/5)  # Chirp mass in units of kilograms 
    f_max = final_frequency(M_chirp,eta)
    t_max = T_chirp(fmin,M_chirp,eta)
    logMchirp = np.log(M_chirp)
    eps_GR = 0
    eps_AP = (4*1*1e-2)
    delta_t = 1/(2*f_max)
    t = np.arange(0,t_max,delta_t)     
    n_t = len(t)  
    delta_f = 1/(n_t*delta_t)  
    freq_bin = np.arange(fmin,f_max,delta_f) 
    n_f = len(freq_bin)  
    pars = [logMchirp,eta,beta,Deff]
    h_f = htilde(freq_bin,eps_GR,pars)
    return freq_bin,h_f
    

    
