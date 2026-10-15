from pathlib import Path
P=Path(__file__).resolve().parent
s=(P/'conditional_source_view.py').read_text()
prefix=s[:s.index('fig,axes=')]
exec(prefix)
F=R/'FillingGap/results_prd_tail_20261015';T=F/'fresh_test'
metrics=np.load(P.parent/'ringdown_frequency_examples_20261015/publication_metrics.npz')
pars_all=np.load(T/'parameters.npy',mmap_mode='r')
ft=np.array([get_lm_f0tau(float(p[9]),float(p[10]),2,2) for p in pars_all]);f=ft[:,0]
good=(metrics['snr']>=10)&(metrics['mismatch']<.005)&(metrics['gap_relative']<.15)
low=np.flatnonzero(good&(f>.006)&(f<.009)&(metrics['coverage']>3))
high=np.flatnonzero(good&(f>.035))
indices=list(sorted(low,key=lambda i:metrics['mismatch'][i])[:8])+list(sorted(high,key=lambda i:-f[i])[:8])
truth=np.load(T/'truth.npy',mmap_mode='r');noise=np.load(T/'noise.npy',mmap_mode='r');pred=np.load(F/'variable_optimized_prediction.npy',mmap_mode='r')
asd=np.sqrt(psd);asd[:,0]=asd[:,1];asd[:,-1]=asd[:,-2]
tt=np.arange(1056)*.5
mask=np.array([(tt>=metrics['start'][i])&(tt<metrics['start'][i]+metrics['length'][i]) for i in indices])
data=dict(index=np.array(indices),seeds=metrics['seeds'][indices],truth=np.array(truth[indices],float),noise=np.array(noise[indices],float),reconstruction=np.fft.irfft(np.fft.rfft(np.array(pred[indices],float))*asd,n=1056),mask=mask,time=tt)
data['observation']=data['truth']*(1-mask[:,None,:])+data['noise']
reports=[];saved={}
body=s[s.index('for j,(idx,seed)'):s.index(' exponent=int')]
body=body.replace('reports.append(report);print(json.dumps(report),flush=True)',"report.update(direct_mismatch=float(metrics['mismatch'][idx]),snr=float(metrics['snr'][idx]),gap_relative=float(metrics['gap_relative'][idx]));reports.append(report);print(json.dumps(report),flush=True)\n np.savez_compressed(P/f'candidate_{idx}.npz',time=ts,source=source,reconstruction=source_rec,projected_noise=source_noise,conditional_observation_fit=source_baseline,response_basis=tdi_basis,truth=data['truth'][j],observation=data['observation'][j],noise=data['noise'][j],ae_reconstruction=data['reconstruction'][j],mask=data['mask'][j])")
exec(body)
(P/'frequency_screen.json').write_text(json.dumps(reports,indent=2))
