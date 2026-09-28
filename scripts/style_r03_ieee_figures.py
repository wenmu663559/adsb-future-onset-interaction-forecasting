"""Restyle frozen numerical evidence at final IEEE figure width; no experiments."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/figures'
records=json.loads((ROOT/'reports/references/r03_review_group_uncertainty_2026-09-04.json').read_text())['records']
diag=json.loads((ROOT/'reports/references/r03_review_diagnostics_2026-09-04.json').read_text())['by_airport']
plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42})
for airport,stem in [('KBTP','figure_1_kbtp_confirmation'),('KAGC','figure_2_kagc_transfer')]:
    rs={(r['horizon'],r['comparison']):r for r in records if r['airport']==airport}
    hs=(30,120,300);xx=np.arange(3)
    fig,axs=plt.subplots(1,2,figsize=(7.16,2.55),layout='constrained')
    for off,key,label,col in [(-.24,'candidate_mean','Pair geometry B','#8ab3d7'),(0,'baseline_mean','Aggregate A','#c4d1de')]:
        axs[0].bar(xx+off,[rs[h,'count_B_vs_A'][key] for h in hs],width=.23,label=label,color=col)
    axs[0].bar(xx+.24,[rs[h,'count_B_vs_pair_rate']['baseline_mean'] for h in hs],width=.23,label='Frozen pair rate',color='#eed4ad')
    prob=[r for r in records if r['airport']==airport and r['comparison'].startswith('prob')]
    pr={r['horizon']:r for r in prob}
    axs[1].bar(xx-.14,[pr[h]['candidate_mean'] for h in hs],width=.27,label='B probability',color='#8ab3d7')
    axs[1].bar(xx+.14,[pr[h]['baseline_mean'] for h in hs],width=.27,label='Frozen prevalence',color='#eed4ad')
    for ax,title,ylabel in [(axs[0],'(a) Future-onset episode count','Equal-group mean absolute error'),(axs[1],'(b) Threshold-exceedance probability','Equal-group Brier score')]:
        ax.set_title(title);ax.set_ylabel(ylabel);ax.set_xticks(xx,[str(h) for h in hs]);ax.set_xlabel('Forecast horizon (s)')
        ax.set_ylim(0,ax.get_ylim()[1]*1.30);ax.legend(loc='upper left',frameon=False,ncol=1)
        ax.spines[['top','right']].set_visible(False)
    for ext in ('png','pdf'):fig.savefig(OUT/f'{stem}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(7.16,4.6),layout='constrained')
for ai,a in enumerate(('KBTP','KAGC')):
    for hi,h in enumerate((30,120,300)):
        ax=axs[ai,hi];bs=[b for b in diag[a][str(h)]['reliability_pooled'] if b['n']]
        ax.plot([0,1],[0,1],color='#aaaaaa',ls='--',lw=.7)
        ax.plot([b['mean_probability'] for b in bs],[b['observed_rate'] for b in bs],color='#5c8caf',marker='o',ms=3,lw=.8)
        for bi,b in enumerate(bs):ax.annotate(str(b['n']),(b['mean_probability'],b['observed_rate']),xytext=(2,5 if bi%2==0 else -10),textcoords='offset points',fontsize=8)
        ax.set(xlim=(-.04,1.07),ylim=(-.04,1.10),title=f'{a}, {h} s',xlabel='Mean predicted probability',ylabel='Observed event fraction')
        ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1]);ax.spines[['top','right']].set_visible(False)
for ext in ('png','pdf'):fig.savefig(OUT/f'figure_3_reliability.{ext}',dpi=300,facecolor='white')
print('Three IEEE-width figures exported; inputs unchanged.')
