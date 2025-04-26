import numpy as np
import matplotlib.pyplot as plt
Ms = 1.99*10**30
pc = 3.26*9.46*10**15
c = 3.0e8  
cG = 6.67430e-11 

# M = 3e5*Ms          # 基础质量 [kg]
# dM =0                 
#                     # 光速 [m/s]
# r = 0.5e9*pc               # 距离 [m]
# dr = 0  
# a=0.9               # 自旋参数
# v=2/9               #对称质量比    
# phi = 0                # φ（弧度）    
# x = np.pi/3  # 弧度     
# theta =0  # 观察方位角 θ
# psi = np.pi/3    # 偏振角 ψ
# phi0 = np.pi    # 极化角 φ₀
def A1(v):
    return 0.864*v

def  A3(v):
    return 0.44*(1 - 4*v)**0.45*A1(v)
mode_matrix = np.array([
    [0.6000, -0.2339, 0.4175, -0.3000, 2.3561, -0.2277],
    [1.5251, -1.1568, 0.1292,  0.7000, 1.4187, -0.4990],
    [1.8956, -1.3043, 0.1818,  0.9000, 2.3430, -0.4810],
    [2.3000, -1.5056, 0.2244,  1.1929, 3.1191, -0.4825],
])

# 对应模态标签
modes = [(2, 1), (2, 2), (3, 3), (4, 4)]

# 封装返回关于 M, chi 的 QNM 函数
def get_qnm_functions(mode_matrix, modes):
    """
    为每个模式生成一个返回 ω_GR 和 τ_GR 的函数，保存在字典中
    """
    qnm_functions = {}

    for i, (l, m) in enumerate(modes):
        f1, f2, f3, q1, q2, q3 = mode_matrix[i]

        def qnm_func(M, chi, f1=f1, f2=f2, f3=f3, q1=q1, q2=q2, q3=q3):
            omega = (f1 + f2 * (1 - chi)**f3) / (M)/cG*c**3
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




# ----------------------------
# 模式参数：对两个模式(1)和(3)
# ----------------------------
# 振幅因子 A
# A = {1: A1(v), 3: A3(v)}
# # 阻尼时间常数 τ
# tau = {1: tau22(M,a), 3: tau33(M,a)}
# # 角频率 w（单位：rad/s），例如各模式的频率
# w = {1: w22(M,a), 3: w33(M,a)}
# # 模式指数 m
# m = {1: 2, 3: 3}

# ----------------------------
# 球谐函数部分，对应 Yp 和 Yc
# 这里 x 为一个标量（角度，单位：弧度），通常由物理模型或观测给出
# ----------------------------

# Yp = {}
# Yc = {}
# Yp[1] = np.sqrt(5/(4*np.pi)) * (1 + np.cos(x)**2) / 2.0
# Yp[3] = - np.sqrt(21/(8*np.pi)) * (1 + np.cos(x)**2) / 2.0 * np.sin(x)
# Yc[1] = np.sqrt(5/(4*np.pi)) * np.cos(x)
# Yc[3] = - np.sqrt(21/(8*np.pi)) * np.cos(x) * np.sin(x)

# ----------------------------
# 计算模式 (1) 和 (3) 对应的 h₊ 和 hₓ 分量，使用 numpy 的向量化计算
# ----------------------------
# 公共前置因子（注意所有模式共享此因子）
#prefactor = cG * (M * (1 + dM)) / (c**2 * r * (1 + dr))

# ----------------------------
# 时间参数，t为序列输入
# ----------------------------
# t0 = 0.                 # 参考时间 [s]
# T = 500.0                 # 总时长 [s]
# num_points = 1000         # 时间序列点数
# t = np.linspace(t0, t0 + T, num_points)  # t为 numpy 数组
# # 模式 (1)：
# hp1 = prefactor * A[1] * np.exp(-(t - t0)/tau[1]) * Yp[1] * np.cos(w[1]*(t - t0) - m[1]*phi)
# hc1 = -prefactor * A[1] * np.exp(-(t - t0)/tau[1]) * Yc[1] * np.sin(w[1]*(t - t0) - m[1]*phi)

# # 模式 (3)：
# hp3 = prefactor * A[3] * np.exp(-(t - t0)/tau[3]) * Yp[3] * np.cos(w[3]*(t - t0) - m[3]*phi)
# hc3 = -prefactor * A[3] * np.exp(-(t - t0)/tau[3]) * Yc[3] * np.sin(w[3]*(t - t0) - m[3]*phi)

# # 假设总的 h₊ 和 hₓ 分量为各模式的线性叠加
# hpt = hp1 + hp3  # plus 极
# hct = hc1 + hc3  # cross 极

# ----------------------------
# 计算天线图样因子，并合成最终时域信号 h_st
# 公式：
#   h_st = h_pt * [0.5*(1+cos²θ)*cos(2ψ)*cos(2φ₀) - cosθ*sin(2ψ)*sin(2φ₀)]
#        + h_ct * [0.5*(1+cos²θ)*sin(2ψ)*cos(2φ₀) + cosθ*cos(2ψ)*sin(2φ₀)]
# 注意：θ, ψ, φ₀ 均为常数（弧度），φ₀ 这里表示源的极化角，不是初相位。
# ----------------------------


# antenna_plus = 0.5*(1 + np.cos(theta)**2)*np.cos(2*psi)*np.cos(2*phi0) - np.cos(theta)*np.sin(2*psi)*np.sin(2*phi0)
# antenna_cross = 0.5*(1 + np.cos(theta)**2)*np.sin(2*psi)*np.cos(2*phi0) + np.cos(theta)*np.cos(2*psi)*np.sin(2*phi0)

# hst = hpt * antenna_plus + hct * antenna_cross

def hst(time_array,M_tot,DL,a,v,phi=0,x=np.pi,theta=0,psi=np.pi/3,phi0=np.pi):
    """
    计算 hst 的函数
    time_array: 时间序列
    M: 质量 Ms
    r: 距离 Gpc
    a: 自旋参数
    v: 对称质量比
    phi, x, theta, psi, phi0: 模式参数
    返回 hst 数组
    """
    
    M = M_tot*Ms          
    r = DL*1e9*pc               
    dM =0               
    dr = 0  
    # 模式参数：对两个模式(1)和(3)  即22和33
    # 振幅因子 A
    A = {1: A1(v), 3: A3(v)}
    # 阻尼时间常数 τ
    tau = {1: tau22(M,a), 3: tau33(M,a)}
    # 角频率 w（单位：rad/s），例如各模式的频率
    w = {1: w22(M,a), 3: w33(M,a)}
    # 模式指数 m
    m = {1: 2, 3: 3}
    # ----------------------------
    # 球谐函数部分，对应 Yp 和 Yc
    Yp = {}
    Yc = {}
    Yp[1] = np.sqrt(5/(4*np.pi)) * (1 + np.cos(x)**2) / 2.0
    Yp[3] = - np.sqrt(21/(8*np.pi)) * (1 + np.cos(x)**2) / 2.0 * np.sin(x)
    Yc[1] = np.sqrt(5/(4*np.pi)) * np.cos(x)
    Yc[3] = - np.sqrt(21/(8*np.pi)) * np.cos(x) * np.sin(x)
    # ----------------------------
    prefactor = cG * (M * (1 + dM)) / (c**2 * r * (1 + dr))   
    # ----------------------------
    # 时间参数，t为序列输入
    t= time_array
    # ----------------------------
    t0 = 0.                 # 参考时间 [s]
    hp1 = prefactor * A[1] * np.exp(-(t - t0)/tau[1]) * Yp[1] * np.cos(w[1]*(t - t0) - m[1]*phi)
    hc1 = -prefactor * A[1] * np.exp(-(t - t0)/tau[1]) * Yc[1] * np.sin(w[1]*(t - t0) - m[1]*phi)

    # 模式 (3)：
    hp3 = prefactor * A[3] * np.exp(-(t - t0)/tau[3]) * Yp[3] * np.cos(w[3]*(t - t0) - m[3]*phi)
    hc3 = -prefactor * A[3] * np.exp(-(t - t0)/tau[3]) * Yc[3] * np.sin(w[3]*(t - t0) - m[3]*phi)

    # 假设总的 h₊ 和 hₓ 分量为各模式的线性叠加
    hpt = hp1 + hp3  # plus 极
    hct = hc1 + hc3  # cross 极

    # ----------------------------
    # 计算天线图样因子，并合成最终时域信号 h_st
    # 公式：
    #   h_st = h_pt * [0.5*(1+cos²θ)*cos(2ψ)*cos(2φ₀) - cosθ*sin(2ψ)*sin(2φ₀)]
    #        + h_ct * [0.5*(1+cos²θ)*sin(2ψ)*cos(2φ₀) + cosθ*cos(2ψ)*sin(2φ₀)]
    # 注意：θ, ψ, φ₀ 均为常数（弧度），φ₀ 这里表示源的极化角，不是初相位。
    # ----------------------------


    antenna_plus = 0.5*(1 + np.cos(theta)**2)*np.cos(2*psi)*np.cos(2*phi0) - np.cos(theta)*np.sin(2*psi)*np.sin(2*phi0)
    antenna_cross = 0.5*(1 + np.cos(theta)**2)*np.sin(2*psi)*np.cos(2*phi0) + np.cos(theta)*np.cos(2*psi)*np.sin(2*phi0)

    hst = hpt * antenna_plus + hct * antenna_cross
    return hst