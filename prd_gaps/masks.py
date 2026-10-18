"""Availability propagation for the equal-arm fastlisaresponse TDI operator."""
import numpy as np
from scipy.ndimage import minimum_filter1d

def backend_terms(response):
    """Read the X terms from arrays passed to the backend, including old releases
    whose tdi_combinations property setter does not retain its input list.
    """
    operations = np.asarray(response.tdi_operation_index)
    delays = np.asarray(response.tdi_link_combinations)
    terms = []
    for operation in np.flatnonzero(np.asarray(response.channels) == 0):
        delay_links = delays[operations == operation]
        terms.append(dict(link=int(response.tdi_base_links[operation]),
                          links_for_delay=delay_links[delay_links > 0].tolist(),
                          sign=float(response.tdi_signs[operation])))
    if len(terms) != 16:
        raise ValueError('Expected 16 X terms for second-generation Michelson TDI')
    return terms

def erode_valid(valid, radius):
    return minimum_filter1d(np.asarray(valid, dtype=np.uint8), size=2*radius+1,
                            axis=-1, mode='constant', cval=0).astype(bool)

def propagate_tdi(link_valid, links, combinations, fs, arm_seconds, order=25):
    """Conservative stencil union; use with equal-arm orbits only.

    Each term reads one link at the summed delay. An order+2 sample guard
    encloses the fractional-delay interpolation stencil. The backend's actual
    term list supplies the delays. Backend replacement tests validate coverage.
    """
    valid = np.asarray(link_valid, bool)
    if valid.ndim != 2 or valid.shape[0] != 6 or len(links) != 6:
        raise ValueError('Expected six directed link availability streams')
    index = {int(link): i for i, link in enumerate(links)}
    guarded = erode_valid(valid, order+2)
    n = valid.shape[-1]
    xyz = np.ones((3, n), bool)
    t = np.arange(n)
    for c in range(3):
        for term in combinations:
            i, j = divmod(int(term['link']), 10)
            link = 10*((i-1+c)%3+1)+((j-1+c)%3+1)
            delay = len(term['links_for_delay'])*arm_seconds*fs
            lo = np.floor(t-delay).astype(int)
            hi = np.ceil(t-delay).astype(int)
            inside = (lo >= 0) & (hi < n)
            xyz[c] &= inside & guarded[index[link], np.clip(lo, 0, n-1)] \
                              & guarded[index[link], np.clip(hi, 0, n-1)]
    ae = np.stack((xyz[2] & xyz[0], xyz.all(axis=0)))
    return xyz, ae

def edge_taper(valid, width):
    """Half-cosine weights on available samples; gaps and endpoints remain zero."""
    from scipy.ndimage import distance_transform_edt
    valid = np.asarray(valid, bool)
    out = np.zeros(valid.shape, float)
    for row, dest in zip(valid.reshape(-1, valid.shape[-1]), out.reshape(-1, valid.shape[-1])):
        distance = distance_transform_edt(np.pad(row, 1))[1:-1]
        dest[:] = .5*(1-np.cos(np.pi*np.minimum(distance/max(width, 1), 1)))
    return out
