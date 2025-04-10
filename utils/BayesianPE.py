import sys
import os

# 将 `code/Ringdown_gap_filling/Proj/` 添加到 sys.path
proj_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if proj_dir not in sys.path:
    sys.path.append(proj_dir)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import bilby
import corner
from config.config import Config
from data.waveform import *
from data.ringdown_waveform import Gap_dir as Ga


def model(time,Mtot,M_ratio,R_shift,**kwargs):
    para=[Mtot,M_ratio,R_shift]
    freq_ifft = np.arange(Config.f_in, Config.f_out, Config.f_step)
    sf22, sf21,sf33, sf44 = sf_decomposition(freq_ifft, para, para_dw, para_dtau)
    sf = sf22 + sf21 + sf33 + sf44
    output= Ga.Freq_ifft(sf)
    if np.any(np.iscomplex(output)):
        #print(f"Complex model output detected: {output}")
        output=np.real(output)
    return output


Mtot = np.random.uniform(Config.parameters[0], Config.parameters[1])
M_ratio = np.random.uniform(Config.parameters[2], Config.parameters[3])
R_shift = np.random.uniform(Config.parameters[4], Config.parameters[5])
signal_length = Config.signal_length

para1 = [Mtot, M_ratio, R_shift]
freq_ifft = np.arange(Config.f_in, Config.f_out, Config.f_step)
f_sf22, f_sf21,f_sf33, f_sf44 = sf_decomposition(freq_ifft, para1, para_dw, para_dtau)
f_sf = f_sf22 + f_sf21 + f_sf33 + f_sf44
st = Ga.Freq_ifft(f_sf)

duration = 1 / Config.f_step 



n_points = len(freq_ifft) * 2  


# 时间轴
time = np.linspace(0, duration, n_points, endpoint=False)
yobs=st.real

# Example of using bilby.core.likelihood functions
# Define a new likelihood function using bilby.core.likelihood

class CustomLikelihood(bilby.Likelihood):
    def __init__(self, time, yobs, model):
        super().__init__(parameters=dict())
        self.time = time
        self.yobs = yobs
        self.model = model
        self.sigma = 0.1

    def log_likelihood(self):
        prediction = self.model(self.time, **self.parameters)
        residual = self.yobs - prediction
        log_l = -0.5 * (np.sum((residual / self.sigma) ** 2) + len(self.yobs) * np.log(2 * np.pi * self.sigma ** 2))
        if np.iscomplex(log_l):
            print(f"Complex log likelihood: {log_l}")
        return log_l.real  # 如果是复数，取实部返回

# Example usage
custom_likelihood = CustomLikelihood(time, yobs, model)
# priors = dict(
#     Mtot=bilby.core.prior.Uniform(Config.parameters[0], Config.parameters[1]),
#     M_ratio=bilby.core.prior.Uniform(Config.parameters[2], Config.parameters[3]),
#     R_shift=bilby.core.prior.Uniform(Config.parameters[4], Config.parameters[5])
# )
from bilby.core.prior import PriorDict

priors = PriorDict()
priors['Mtot'] = bilby.core.prior.Uniform(Config.parameters[0], Config.parameters[1])
priors['M_ratio'] = bilby.core.prior.Uniform(Config.parameters[2], Config.parameters[3])
priors['R_shift'] = bilby.core.prior.Uniform(Config.parameters[4], Config.parameters[5])

from bilby.bilby_mcmc.proposals import AdaptiveGaussianProposal, DifferentialEvolutionProposal, UniformProposal, KDEProposal

from bilby.bilby_mcmc.proposals import ProposalCycle

adaptive_gaussian_proposal = AdaptiveGaussianProposal(priors=priors)
differential_evolution_proposal = DifferentialEvolutionProposal(priors=priors)
uniform_proposal = UniformProposal(priors=priors)
kde_proposal = KDEProposal(priors=priors)

# 构建 ProposalCycle
proposal_cycle = ProposalCycle([
    adaptive_gaussian_proposal,
    differential_evolution_proposal,
    uniform_proposal,
    kde_proposal
])



result = bilby.run_sampler(
    likelihood=custom_likelihood,
    priors=priors,
    label="custom_model",
    outdir="custom_model_output/result3",
    sampler="bilby_mcmc",
    nsamples=500,
    proposal_cycle=proposal_cycle,
    printdt=20,
    L1steps=1,
)
print(Mtot,M_ratio,R_shift)