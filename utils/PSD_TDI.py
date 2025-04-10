import numpy as np
# from TDI.Constants import *
# from TDI.FFTTools import *
from config.config import Config
C=2.9979246*1e8
class PSD_y(): #PSD in the fractional frequency difference unit
    def __init__(self, sacc = Config.Sacc, sopt = Config.Sopt, L = Config.L): # default Taiji [sacc] = acceleration, [sopt] = distance
        self.sa = sacc
        self.so = sopt
        self.L = L

    def PSD_Sa(self, f): 
        u = 2. * np.pi * f * self.L / C
        return (self.sa * self.L / u / C ** 2) ** 2 * (1. + (0.4e-3 / f) ** 2) * (1. + (f / 8e-3) ** 4)

    def PSD_So(self, f):
        u = 2. * np.pi * f * self.L / C
        return (u * self.so / self.L) ** 2 * (1. + (2e-3 / f) ** 4)

    def PSD_X(self, f): # self.sa, self.so are the asd of acc and opt noise, WTN and wang and Vallisneri
        u = 2. * np.pi * f * self.L / C
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        return Sa * (8. * (np.sin(2. * u)) ** 2 + 32. * (np.sin(u)) ** 2) \
                + 16. * So * (np.sin(u)) ** 2

    def PSD_A(self, f): # WTN
        u = 2. * np.pi * f * self.L / C
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        return 8. * So * (2. + np.cos(u)) * (np.sin(u)) ** 2 \
                + 16. * Sa * (3. + 2. * np.cos(u) + np.cos(2. * u)) * (np.sin(u)) ** 2

    def PSD_T(self, f): # WTN
        u = 2. * np.pi * f * self.L / C
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        return 16. * So * (1. - np.cos(u)) * (np.sin(u)) ** 2 \
                + 128. * Sa * (np.sin(u)) ** 2 * (np.sin(u / 2.)) ** 4
    
    def PSD_T_unequal(self, f, L1, L2, L3):
        u1 = 2. * np.pi * f * L1 / C
        u2 = 2. * np.pi * f * L2 / C
        u3 = 2. * np.pi * f * L3 / C
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        Ocoef = (np.sin(u3)) ** 2 + (np.sin(u2)) ** 2 + (np.sin(u1)) ** 2 \
                - np.sin(u3) * np.sin(u2) * np.cos(u1) * np.cos(u3 - u2) \
                - np.sin(u3) * np.sin(u1) * np.cos(u2) * np.cos(u3 - u1) \
                - np.sin(u2) * np.sin(u1) * np.cos(u3) * np.cos(u2 - u1)
        Acoef = 3. - (np.cos(u3) * np.cos(u2)) ** 2 - (np.cos(u3) * np.cos(u1)) ** 2 - (np.cos(u2) * np.cos(u1)) ** 2 \
                - 2. * np.sin(u3) * np.sin(u2) * np.cos(u1) * np.cos(u3 - u2) \
                - 2. * np.sin(u3) * np.sin(u1) * np.cos(u2) * np.cos(u3 - u1) \
                - 2. * np.sin(u2) * np.sin(u1) * np.cos(u3) * np.cos(u2 - u1)
        return 16. / 3. * So * Ocoef + 32. / 3. * Sa * Acoef
    
    def PSD_alpha(self, f): # wang, Vallis
        u = 2. * np.pi * f * self.L / C
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        # equivalent
    #     return 8. * Sa * (5. + 4. * np.cos(u) + 2. * np.cos(2. * u)) * (np.sin(u / 2.)) ** 2 \
    #         + 6. * So
        return 8. * Sa * ((np.sin(1.5 * u)) ** 2 + 2. * (np.sin(u)) ** 2) \
                + 6. * So

    def PSD_alpha_beta(self, f): # Vallis
        u = 2. * np.pi * f * self.L / C
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        return So * 2. * (2. * np.cos(u) + np.cos(2. * u)) + Sa * 4. * (np.cos(u) - 1.)
    



# def SNR(t_arr, sig_arr, psd_func):
#     fsample = 1. / (t_arr[1] - t_arr[0])
#     f, sigf = FFT_complex(sig_arr, fsample=fsample)
#     snr2 = 0
#     for i in range(len(f)):
#         snr2 += (np.abs(sigf[i])) ** 2 / psd_func(f[i])
#     snr2 *= 4. * (f[1] - f[0])
#     return np.sqrt(snr2)
