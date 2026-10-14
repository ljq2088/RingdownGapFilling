"""Preserve the design of the result figures while presenting the final estimator."""
from pathlib import Path

p = Path(__file__).resolve().parent
s = (p/'make_results_presentation.py').read_text(encoding='utf-8')
s = s.replace("COLORS={'baseline':'#96a2b1','previous':'#3478ae','augmented':'#c66a22','interpolation':'#29988b'}", "COLORS={'augmented':'#c66a22','interpolation':'#29988b'}")
s = s.replace("'augmented':'Augmented refinement'", "'augmented':'DGF'")
s = s.replace("heading+f'\\nMean: {old:.5f} → {new:.5f}'", "heading+f'\\nDGF mean mismatch: {new:.5f}'")
s = s.replace("ncol=4", "ncol=2")
s = s.replace("ax.fill_between(f,po,pa,color='#f2c796',alpha=.34,lw=0)", "")
s = s.replace("ax.loglog(f,po,color=COLORS['baseline'],lw=1.35,label='Original DGF error')", "")
s = s.replace("label='Augmented error'", "label='DGF waveform residual'")
s = s.replace("for name in ['baseline','augmented']", "for name in ['augmented']")
s = s.replace("for name in ['baseline','previous','augmented']:", "for name in ['augmented']:")
s = s.replace("f'(a) Full-population phase accuracy\\nMean: {old:.2f}° → {new:.2f}°'", "f'(a) Full-population phase accuracy\\nDGF mean absolute error: {new:.2f}°'")
s = s.replace("'means':means", "'figure_scope':'Final DGF estimator and interpolation reference; no checkpoint progression', 'means':means")
(p/'make_final_results_figures.py').write_text(s, encoding='utf-8')

s = (p/'make_best_examples.py').read_text(encoding='utf-8')
s = s.replace("    ax.plot(t,base/scale,color='#96a2b1',lw=1.1,ls=':',label='Original DGF',zorder=2)", "")
s = s.replace("label='Augmented refinement'", "label='DGF reconstruction'")
s = s.replace('all three plotted waveforms per panel', 'the target and DGF prediction per panel')
(p/'make_final_waveform_examples.py').write_text(s, encoding='utf-8')
print('Prepared final-estimator plotting scripts with unchanged metric and prediction inputs.')
