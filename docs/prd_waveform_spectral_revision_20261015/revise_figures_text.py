from pathlib import Path
import zipfile,json,hashlib,difflib
P=Path(__file__).resolve().parent
with zipfile.ZipFile(P/'overleaf_before.zip') as z:
    original=z.read('DGF.tex').decode('utf-8')
    # Preserve the unchanged population and method figures byte for byte.
    for n in ['latest_mismatch.pdf','latest_snr_gap.pdf']:
        (P/'source'/n).write_bytes(z.read(n))
s=original
def replace(old,new):
    global s
    assert old in s,old[:100]
    s=s.replace(old,new)
replace('Both channels are expressed in fractional-frequency units.', 'Both channels are expressed in fractional-frequency units. The delayed link combinations redistribute the onset and relative mode contributions in each channel.')
replace('These results demonstrate recovery of damped oscillations through intervals containing the signal peak in the simulated source population.', 'These results demonstrate recovery of damped oscillations across data gaps in the simulated source population.')
start=s.index('Figure~\\ref{fig:examples} shows two')
end=s.index('\\begin{figure*}',start)
s=s[:start]+r'''Figure~\ref{fig:examples} shows two $\rho\ge10$ sources reconstructed across 47- and 35-s gaps, with noise-weighted mismatches $7.22\times10^{-4}$ and $7.88\times10^{-4}$. The gaps contain 41.9\% and 17.2\% of the respective joint signal squared norms. Their gap-local relative errors are 5.3\% and 5.5\%. The full observations show the noise throughout each record, while the expanded waveform panels resolve the oscillations across the gap and the subsequent decay. The windows extend for 6.2 and 7.5 dominant-mode damping times after the joint envelope peak.
''' +s[end:]
start=s.index('\\caption{Waveform reconstruction for two')
end=s.index('\\includegraphics[width=0.885\\textwidth]{latest_phase.pdf}',start)
s=s[:start]+r'''\caption{Waveform reconstruction for two $\rho\ge10$ sources across 47- and 35-s gaps in the $A$ (left) and $E$ (right) channels. In each labeled panel, the upper axes show the noisy observation $d=Gh+n$ (gray), clean signal $h$, gapped signal $Gh$ and DGF reconstruction $\widehat h$ over the full 528-s record. The lower axes expand the signal and reconstruction over the indicated time interval. Shading marks signal suppression, with noise retained in the observation. All curves in both channels share the scale $H_{\rm TDI}=\max_{c,j}|h_{c,j}|$ and connect the original 0.5-s samples. Source seeds are 2042202428 and 2042204884. Appendix~\ref{app:model_analysis} gives the example criteria.}\label{fig:examples}\par\vspace{8pt}
''' +s[end:]
replace('2042203322','2042202428')
replace(r'\includegraphics[width=0.885\textwidth]{latest_examples.pdf}',r'\includegraphics[width=0.95\textwidth]{latest_examples.pdf}')
replace('Figure~\\ref{fig:psd} compares population-averaged target, residual and instrument-noise spectra.',r'Figure~\ref{fig:psd} compares the error spectra before and after reconstruction. The input error is $e_{\rm in}=d-h=n-mh$, and the reconstruction error is $e_{\rm out}=\widehat h-h$. Their population-averaged spectra measure the combined effect of denoising and gap reconstruction. The reconstructed error power is lower at every displayed Fourier bin in both channels; the lower panels give the ratio to the input error power.')
start=s.index('\\caption{Mean power spectral densities of the target signal')
end=s.index('\\label{fig:psd}',start)
s=s[:start]+r'''\caption{Error spectra before and after reconstruction over all 6,000 test sources. Upper panels: mean periodograms of the input error $e_{\rm in}=d-h$, reconstruction error $e_{\rm out}=\widehat h-h$ and instrument noise $n$ in the $A/E$ channels. Lower panels: $10\log_{10}[\overline P_{e_{\rm out}}(f)/\overline P_{e_{\rm in}}(f)]$, with zero marking equal error power. Each periodogram uses a rectangular window, constant detrending and one-sided density normalization; the bar denotes an arithmetic mean over sources. Only interior Fourier bins are shown. Fractional-frequency PSDs have units $\mathrm{Hz}^{-1}$.}''' +s[end:]
replace('The waveform comparisons show recovery of the decaying oscillations across intervals containing the signal peak, while gap-local and spectral errors quantify amplitude-sensitive deviations.', 'The waveform comparisons show recovery of the oscillations across the gaps and their subsequent decay, while gap-local and spectral errors quantify amplitude-sensitive deviations.')
start=s.index('The examples in Fig.~\\ref{fig:examples} are chosen')
end=s.index('\n',start)
s=s[:start]+r'''The examples in Fig.~\ref{fig:examples} have resolved decaying oscillations and small waveform residuals. Both have $\rho\ge10$, more than six $\tau_{220}$ of post-peak coverage and at least seven resolved extrema per channel. Their gap-local relative errors are below 6\% and their channel peak-phase errors below $1.6^\circ$. The source seeds identify the displayed cases; population statistics include every test source.''' +s[end:]
replace('The target, residual and noise series receive identical processing.',r'The input error $e_{\rm in}=n-mh$, reconstruction error $e_{\rm out}=\widehat h-h$ and instrument noise receive identical processing. The error-power ratio uses the two population-mean periodograms, $10\log_{10}(\overline P_{e_{\rm out}}/\overline P_{e_{\rm in}})$.')
assert all(x not in s for x in ['2042203322','2042200886','40- and 48-s','population-averaged target','target signal $h$, reconstruction residual'])
(P/'source/DGF.tex').write_text(s,encoding='utf-8',newline='\n')
(P/'manuscript.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='before/DGF.tex',tofile='after/DGF.tex')),encoding='utf-8')
f=P/'source/reproducibility_manifest.json';m=json.loads(f.read_text(encoding='utf-8'))
m['figure_examples'].update(json.loads((P/'selected_examples.json').read_text()))
m['figure_examples']['detail_time_ranges_s']=[[65,350],[0,300]]
m['spectral_error_comparison']=json.loads((P/'spectral_audit.json').read_text())
m['spectral_error_comparison']['ratio_definition']='10 log10(mean output-error periodogram / mean input-error periodogram), not mean of source-wise ratios'
for n in m['figure_sha256']:
    m['figure_sha256'][n]=hashlib.sha256((P/'source'/n).read_bytes()).hexdigest()
f.write_text(json.dumps(m,indent=2),encoding='utf-8')
print('Updated figures, example references, spectral definitions and manifest.')
