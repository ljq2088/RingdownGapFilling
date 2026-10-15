from pathlib import Path
import json,hashlib,shutil
P=Path(__file__).resolve().parent;S=P/'source';f=S/'DGF.tex';s=f.read_text(encoding='utf-8')
addition=r'''
\subsection{Conditional reconstruction of incident strain}\label{subsec:incident_strain}
The incident polarizations provide a complementary view of the damped oscillations. For the three sources in Fig.~\ref{fig:examples}, we fix the injected detector-frame remnant mass, spin, sky position, inclination, polarization angle and arrival epoch. The source history follows Sec.~\ref{sec:method}. Let $U$ contain the two phase quadratures of each of the four fundamental modes, evaluated in both strain polarizations, and let $R$ contain their corresponding $A/E$ responses. Using the same calibrated PSD and interior Fourier bins as the mismatch, define the real noise-weighted Fourier map $\mathcal W$. The conditional estimate is
\begin{equation}
\widehat s=P\widehat h,\qquad P=U(\mathcal W R)^+\mathcal W,\label{eq:conditional_strain}
\end{equation}
where $s=(h_+,h_\times)$ and the superscript $+$ denotes the Moore--Penrose inverse. We solve for the eight real mode coefficients with all source parameters held fixed. This projection expresses the DGF output within the specified QNM family.

Figure~\ref{fig:conditional_strain} displays the original observations together with the conditional strain estimates. The gray source-domain curves are $s+Pn$, using the same realized instrument noise and projection as in Eq.~\eqref{eq:conditional_strain}. Their smooth, correlated fluctuations follow from the finite QNM basis. Over $0\le t\le4\tau_{220}$, the joint-polarization relative errors $\|\widehat s-s\|_2/\|s\|_2$ are 2.3\%, 3.0\% and 4.6\%. These examples describe strain recovery at fixed injected parameters; the population results in Table~\ref{tab:results} evaluate the direct $A/E$ outputs.

\begin{figure*}[p]\centering
\includegraphics[width=\textwidth]{conditional_source_strain.pdf}
\caption{Conditional incident-strain reconstruction for the three sources in Fig.~\ref{fig:examples}, with one source per column. The upper two rows show the full $A/E$ observations $d=Gh+n$ in fractional-frequency units of $10^{-18}$; blue shading marks the signal-suppression interval. The lower rows show dimensionless $h_+$ and $h_\times$ in units of $10^{-19}$: injected strain (blue), conditional reconstruction $P\widehat h$ (orange dashed), and strain plus projected instrument noise $s+Pn$ (gray). The gray fill spans $s$ and $s+Pn$ for this single noise realization. All noise amplitudes retain their original scale. Source time is measured from the mode-amplitude reference epoch and divided by the injected $\tau_{220}$, respectively 65.0, 62.6 and 32.1 s. The source panels cover four damping times at the native 0.5-s sampling. Source parameters and the response are fixed as specified in Sec.~\ref{subsec:incident_strain}.}\label{fig:conditional_strain}
\end{figure*}

'''
needle=r'\section{Discussion and Conclusions}'
assert needle in s and r'\label{fig:conditional_strain}' not in s
s=s.replace(needle,addition+needle)
f.write_text(s,encoding='utf-8');shutil.copyfile(P/'conditional_source_strain.pdf',S/'conditional_source_strain.pdf')
f=S/'reproducibility_manifest.json';v=json.loads(f.read_text());v['conditional_source_reconstruction']=json.loads((P/'conditional_source_report.json').read_text());v['figure_sha256']['conditional_source_strain.pdf']=hashlib.sha256((S/'conditional_source_strain.pdf').read_bytes()).hexdigest();f.write_text(json.dumps(v,indent=2)+'\n')
