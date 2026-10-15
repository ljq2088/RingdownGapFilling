from pathlib import Path
import json, hashlib
P=Path(__file__).resolve().parent
S=P/'source'
p=S/'DGF.tex'
s=p.read_text(encoding='utf-8')
old='Measuring these modes is the basis of black hole spectroscopy'
new='The excitation process determines their relative amplitudes and phases~\\cite{Zhang2013Excitation}. Measuring these modes is the basis of black hole spectroscopy'
assert old in s
s=s.replace(old,new,1)
old='The full observations show the noise throughout each record, while the expanded waveform panels resolve the oscillations across the gap and the subsequent decay. The windows extend for 6.2 and 7.5 dominant-mode damping times after the joint envelope peak.'
new='The full observations and expanded waveform panels retain the noise throughout each record. The dominant-mode periods are 59.1 and 60.7 s, with damping times of 60.4 and 62.6 s, respectively. The complete records extend for 6.2 and 7.5 dominant-mode damping times after the joint envelope peak.'
assert old in s
s=s.replace(old,new)
start=s.index('\\caption{Waveform reconstruction for two')
end=s.index('\\label{fig:examples}',start)
s=s[:start]+r'''\caption{Waveform reconstruction for two $\rho\ge10$ sources across 47- and 35-s gaps in the $A$ (left) and $E$ (right) channels. Each labeled panel shows the full 528-s observation above an amplitude and time zoom. Both views include $d=Gh+n$ (gray), the clean signal $h$, gapped signal $Gh$ and DGF reconstruction $\widehat h$. Every ordinate gives fractional frequency in units of $10^{-18}$. The zoom uses the same samples and amplitudes, with noise excursions outside its limits clipped by the axes. Shading marks signal suppression. Curves connect the original 0.5-s samples. Source seeds are 2042202428 and 2042204884; Appendix~\ref{app:model_analysis} gives the example criteria.}'''+s[end:]
start=s.index('\\caption{Absolute values of the clean and reconstructed')
end=s.index('\\label{fig:tdi_log_amplitudes}',start)
s=s[:start]+r'''\caption{Absolute fractional-frequency amplitudes of the observation (gray), clean signal and DGF reconstruction in $A$ (left) and $E$ (right), for the sources in Fig.~\ref{fig:examples}. Shading marks the gap. The logarithmic axis resolves the decaying signal and differences near zero crossings. All curves use the original samples and physical amplitudes.}'''+s[end:]
p.write_text(s,encoding='utf-8')
b=S/'DGF_references.bib'
v=b.read_text(encoding='utf-8')
assert '@article{Zhang2013Excitation' not in v
v+='''
@article{Zhang2013Excitation,
  author = {Zhang, Zhongyang and Berti, Emanuele and Cardoso, Vitor},
  title = {Quasinormal ringing of {Kerr} black holes. {II}. Excitation by particles falling radially with arbitrary energy},
  journal = {Phys. Rev. D},
  volume = {88},
  pages = {044018},
  year = {2013},
  doi = {10.1103/PhysRevD.88.044018},
  eprint = {1305.4306},
  archivePrefix = {arXiv},
  primaryClass = {gr-qc}
}
'''
b.write_text(v,encoding='utf-8')
p=S/'reproducibility_manifest.json'
v=json.loads(p.read_text(encoding='utf-8'))
selected=json.loads((P/'selected_examples.json').read_text())
v['figure_examples'].update(selected)
v['figure_examples']['waveform_origin_audit']=json.loads((P/'waveform_origin_audit.json').read_text())
v['figure_examples']['zoom_limits']={'source_1_time_s':[65,350],'source_2_time_s':[0,300],'fractional_frequency_units':1e-18,'zoom_y_limits':[-1.3,1.3],'full_record_y_limits':'symmetric, 1.06 times each channel maximum absolute observation','observation_in_all_waveform_axes':True}
for name in v['figure_sha256']:
    f=S/name
    if f.exists():v['figure_sha256'][name]=hashlib.sha256(f.read_bytes()).hexdigest()
p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
