"""IMR signal, equal-arm TDI2, stationary secondary noise and propagated gaps.

Six projected GW links determine availability. Complete A/E secondary noise is
drawn from the scirdv1 model, then the same availability masks both signal and
noise. This is an idealized post-TDI dataset with link-propagated availability.
"""
import numpy as np
import lal
import lalsimulation as ls
from scipy.interpolate import CubicSpline
from astropy.cosmology import FlatLambdaCDM
from pycbc.waveform import get_td_waveform
from pycbc.conversions import get_lm_f0tau
from fastlisaresponse import pyResponseTDI
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import A2TDISens, E2TDISens
from .masks import propagate_tdi, backend_terms

EPOCH = 31558149.7632
COSMO = FlatLambdaCDM(H0=67.4, Om0=.315, Tcmb0=0.)
ARM_SECONDS = 2.5e9/lal.C_SI

class RebasedOrbits(EqualArmlengthOrbits):
    def __init__(self, pad):
        self.shift = EPOCH-pad
        super().__init__()
    @property
    def t_base(self):
        return super().t_base-self.shift

def to_ae(xyz):
    x, y, z = np.asarray(xyz)
    return np.stack(((z-x)/np.sqrt(2), (x-2*y+z)/np.sqrt(6)))

def tdi_from_links(response, links):
    """Version-local adapter for fastlisaresponse's explicit-link input.

    Installed release checks a removed link_space_craft_0_in attribute in its
    y_gw argument branch. Set the same validated buffers before the normal call.
    """
    links = np.asarray(links, dtype=float)
    if links.shape != (len(response.response_orbits.LINKS), response.num_pts):
        raise ValueError('Link array shape differs from the response grid')
    if not hasattr(response, 'tdi_start_ind'):
        raise ValueError('Initialize the response time support with get_projections first')
    response.y_gw_flat = response.xp.asarray(links).flatten().copy()
    response.y_gw_length = response.num_pts
    return response.get_tdi_delays()

def draw_noise(psd, fs, n, rng):
    z = (rng.normal(size=psd.shape)+1j*rng.normal(size=psd.shape))*np.sqrt(n*fs*psd/4)
    z[:, 0] = 0
    z[:, -1] = rng.normal(size=2)*np.sqrt(n*fs*psd[:, -1]/2)
    return np.fft.irfft(z, n=n, axis=-1)

def draw_parameters(seed):
    rng = np.random.default_rng(seed)
    return dict(mass_source=10**rng.uniform(5, np.log10(5e6)), q=rng.uniform(.3, 1),
                redshift=rng.uniform(1, 6), inclination=np.arccos(rng.uniform(-1, 1)),
                longitude=rng.uniform(0, 2*np.pi), latitude=np.arcsin(rng.uniform(-1, 1)),
                polarization=rng.uniform(0, np.pi), phase=rng.uniform(0, 2*np.pi))

class Simulator:
    def __init__(self, n=32768, fs=2., pad_seconds=2048., peak_seconds=2048., order=25):
        self.n, self.fs, self.order = int(n), float(fs), int(order)
        self.pad = int(round(pad_seconds*fs)); self.peak_seconds = float(peak_seconds)
        self.parent_n = self.n+2*self.pad
        orbits = RebasedOrbits(pad_seconds)
        orbits.configure(linear_interp_setup=True)
        self.response = pyResponseTDI(fs, self.parent_n, order=order, tdi='2nd generation',
                                     tdi_chan='XYZ', orbits=orbits, force_backend='cpu')
        self.links = np.asarray(self.response.response_orbits.LINKS)
        self.combinations = backend_terms(self.response)
        f = np.fft.rfftfreq(self.parent_n, 1/fs)
        self.psd = np.zeros((2, len(f)))
        for c, sens in enumerate((A2TDISens, E2TDISens)):
            self.psd[c, 1:] = sens.get_Sn(f[1:], model='scirdv1')
        if not np.isfinite(self.psd).all() or (self.psd < 0).any():
            raise ValueError('Invalid noise PSD')

    @property
    def crop(self): return slice(self.pad, self.pad+self.n)

    def noise(self, seed):
        return draw_noise(self.psd, self.fs, self.parent_n, np.random.default_rng(seed))[:, self.crop]

    def calibrate_psd(self, count=128, seed=90000000):
        """Finite-record periodogram expectation from independent noise only."""
        total = np.zeros((2, self.n//2+1))
        for j in range(count):
            total += abs(np.fft.rfft(self.noise(seed+j)))**2/(self.fs*self.n)
        total[:, 1:-1] *= 2
        return total/count

    def signal(self, parameters, approximant='IMRPhenomTHM'):
        p = dict(parameters)
        mass = p['mass_source']; q = p['q']; z = p['redshift']
        m1 = mass/(1+q); m2 = mass-m1
        code, fraction, spin = ls.SimIMREOBFinalMassSpin(m1, m2, [0., 0., 0.], [0., 0., 0.], ls.SEOBNRv4)
        if code != 0: raise RuntimeError('Remnant estimate failed')
        remnant_detector = mass*fraction*(1+z)
        f220, tau220 = get_lm_f0tau(remnant_detector, spin, 2, 2)
        # A guard on temporal coverage, not an assertion that IMR is a single QNM.
        if self.n/self.fs-self.peak_seconds < 600+6*tau220:
            raise ValueError('Record too short for the chosen mass/redshift: enlarge n')
        distance = float(COSMO.luminosity_distance(z).value)
        prehistory = self.pad/self.fs+self.peak_seconds+700
        msec = mass*(1+z)*lal.MTSUN_SI; eta = q/(1+q)**2
        # Keep the phase reference fixed when the start frequency/padding changes.
        # IMRPhenomT otherwise interprets f_ref=0 at f_lower.
        reference_frequency = .02/msec
        flow = (5/(256*eta*(prehistory*1.5))*msec**(-5/3))**(3/8)/np.pi
        for attempt in range(4):
            hp, hc = get_td_waveform(approximant=approximant, mass1=m1*(1+z), mass2=m2*(1+z),
                                    distance=distance, delta_t=1/self.fs, f_lower=flow, f_ref=reference_frequency,
                                    inclination=p['inclination'], coa_phase=p['phase'],
                                    spin1z=0., spin2z=0.)
            plus, cross = np.asarray(hp), np.asarray(hc)
            envelope = plus**2+cross**2
            peak = int(np.argmax(envelope))
            if peak <= 0 or peak >= len(envelope)-1:
                raise RuntimeError('IMR envelope peak must be inside its waveform')
            left, center, right = envelope[peak-1:peak+2]
            offset = .5*(left-right)/(left-2*center+right)
            times = (np.arange(len(plus))-peak-offset)/self.fs
            if -times[0] >= prehistory: break
            flow *= .75
        else: raise RuntimeError('Insufficient IMR prehistory after bounded retries')
        parent_times = np.arange(self.parent_n)/self.fs-(self.pad/self.fs+self.peak_seconds)
        if parent_times[0] < times[0]: raise RuntimeError('IMR starts inside response support')
        tail_ratio = float(np.hypot(plus[-1], cross[-1])/np.max(np.hypot(plus, cross)))
        if parent_times[-1] > times[-1] and tail_ratio > 1e-5:
            raise RuntimeError('IMR terminal amplitude too large for late zero extension')
        # Sub-sample peak alignment and resampling apply to the generated physical
        # source, before detector response; reconstructed outputs remain untouched.
        strain = np.zeros(self.parent_n, complex)
        inside = parent_times <= times[-1]
        strain[inside] = CubicSpline(times, plus+1j*cross, extrapolate=False)(parent_times[inside])
        strain *= np.exp(-2j*p['polarization'])
        self.response.get_projections(strain, p['longitude'], p['latitude'], t0=self.pad/self.fs)
        links = np.asarray(self.response.y_gw).copy()
        ae = to_ae(self.response.get_tdi_delays())[:, self.crop].copy()
        if not np.isfinite(ae).all() or np.linalg.norm(ae) == 0:
            raise RuntimeError('Invalid TDI response')
        metadata = dict(p, approximant=approximant, distance_mpc=distance,
                        remnant_detector_mass_estimate=remnant_detector, remnant_spin_estimate=spin,
                        f220_estimate_hz=float(f220), tau220_estimate_seconds=float(tau220),
                        source_polarization_peak_seconds=self.peak_seconds,
                        detector_energy_peak_index=int(np.argmax(np.sum(ae*ae, axis=0))),
                        lower_frequency_hz=float(flow), prehistory_seconds=float(-times[0]),
                        phase_reference_frequency_hz=float(reference_frequency),
                        terminal_relative_amplitude=tail_ratio,
                        observation='G*(s+n)', tdi='2nd generation', noise='scirdv1',
                        fs_hz=self.fs, n=self.n, padding_seconds=self.pad/self.fs,
                        tdi_mask_guard_samples=self.order+2,
                        units='fractional frequency', foreground=False)
        return ae, links, metadata

    def masks(self, events):
        """Events contain link (e.g. 12 or 'all'), crop-relative start_s and duration_s."""
        link_valid = np.ones((6, self.parent_n), bool)
        for event in events:
            start = self.pad+int(np.floor(event['start_s']*self.fs))
            stop = self.pad+int(np.ceil((event['start_s']+event['duration_s'])*self.fs))
            if start < 0 or stop > self.parent_n or stop <= start:
                raise ValueError('Invalid link event interval')
            rows = np.arange(6) if event['link'] == 'all' else np.flatnonzero(self.links == event['link'])
            if len(rows) == 0: raise ValueError('Unknown directed link')
            link_valid[rows, start:stop] = False
        xyz, ae = propagate_tdi(link_valid, self.links, self.combinations,
                               self.fs, ARM_SECONDS, self.order)
        return link_valid, xyz, ae[:, self.crop]

    def example(self, seed, events, parameters=None):
        signal, links, meta = self.signal(draw_parameters(seed) if parameters is None else parameters)
        noise = self.noise(seed+100000000)
        lv, xyz, valid = self.masks(events)
        observation = np.where(valid, signal+noise, 0)
        meta['events'] = events
        return dict(signal=signal, noise=noise, observation=observation, valid=valid,
                    link_valid=lv, links=links, metadata=meta)
