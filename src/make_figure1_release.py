#!/usr/bin/env python3
"""Figure 1 (release version), drawn from the calibration outputs and exported as vector PDF,
600-dpi PNG and 600-dpi TIFF."""
import numpy as np, pandas as pd, os, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rc_common import OUT_DIR
cal=pd.read_csv(os.path.join(OUT_DIR,'calibration_final.csv'))
keys=[('weak','repick'),('weak','fixed'),('strength','repick'),('strength','fixed')]
labels=['Reshuffle,\nre-selected','Reshuffle,\nfixed','Strength,\nre-selected','Strength,\nfixed']
d={(r['null'],r['hubs']):r for _,r in cal.iterrows()}
fp=[float(d[k]['fp_rate']) for k in keys]; lo=[float(d[k]['ci_lo']) for k in keys]; hi=[float(d[k]['ci_hi']) for k in keys]
rs=np.load(os.path.join(OUT_DIR,'calibration_pseudostudy_r_weak_repick.npy'))
loc=json.load(open(os.path.join(OUT_DIR,'calibration_observed_location.json'))); r_obs=loc['r_obs']; pct=round(100*loc['percentile'])
suffix={1:'st',2:'nd',3:'rd'}.get(pct%10 if pct%100 not in (11,12,13) else 0,'th')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':0.8,
    'axes.edgecolor':'#333333','xtick.major.width':0.8,'ytick.major.width':0.8,
    'xtick.color':'#111111','ytick.color':'#111111','axes.labelcolor':'#111111','text.color':'#111111'})
fig,(axA,axB)=plt.subplots(1,2,figsize=(7.4,3.4),gridspec_kw={'width_ratios':[1.1,1]})

x=np.arange(4); colors=['#2b2b2b','#9a9a9a','#9a9a9a','#9a9a9a']
yerr=np.array([[fp[i]-lo[i] for i in range(4)],[hi[i]-fp[i] for i in range(4)]])
axA.bar(x,fp,width=0.6,color=colors,edgecolor='#111111',linewidth=0.6,zorder=3)
axA.errorbar(x,fp,yerr=yerr,fmt='none',ecolor='#111111',elinewidth=1.0,capsize=3.5,capthick=1.0,zorder=5)
axA.axhline(0.05,ls=(0,(4,3)),color='#111111',linewidth=0.9,zorder=2)
for i in range(4):
    axA.text(i,hi[i]+0.014,f'{fp[i]:.2f}',ha='center',va='bottom',fontsize=9,fontweight='bold' if i==0 else 'normal')
axA.set_xticks(x); axA.set_xticklabels(labels,fontsize=8.5)
axA.set_ylabel('Conditional overcall rate',fontsize=10)
axA.set_ylim(0,0.50)
axA.set_yticks([0,0.05,0.1,0.2,0.3,0.4,0.5]); axA.set_yticklabels(['0','0.05','0.1','0.2','0.3','0.4','0.5'],fontsize=8.5)
axA.spines['top'].set_visible(False); axA.spines['right'].set_visible(False)
axA.annotate('nominal',xy=(0,0.05),xytext=(-0.135,0.05),textcoords=axA.get_yaxis_transform(),
             ha='right',va='center',fontsize=8,color='#111111',style='italic')
axA.text(-0.28,1.05,'A',transform=axA.transAxes,fontweight='bold',fontsize=15,va='bottom',ha='left')

axB.hist(rs,bins=34,color='#cccccc',edgecolor='#8a8a8a',linewidth=0.3,zorder=2)
axB.axvline(0,color='#888888',ls=(0,(3,2)),linewidth=0.8,zorder=1)
ymax_data=axB.get_ylim()[1]
axB.text(0.012, ymax_data*0.55, 'no effect\n(r = 0)', ha='left', va='center', fontsize=8, color='#666666', linespacing=1.3)
axB.axvline(r_obs,color='#b2182b',linewidth=1.7,zorder=4)
axB.text(r_obs+0.03,ymax_data*0.95,f'observed r = {r_obs:.2f}\n({pct}{suffix} percentile)',ha='left',va='top',
         fontsize=8.5,color='#b2182b',fontweight='bold',linespacing=1.3)
axB.set_xlabel('Rank-biserial effect size, r',fontsize=10)
axB.set_ylabel('Number of pseudo-studies',fontsize=10)
axB.set_xlim(-0.8,0.25); axB.tick_params(labelsize=8.5)
axB.spines['top'].set_visible(False); axB.spines['right'].set_visible(False)
axB.text(-0.28,1.05,'B',transform=axB.transAxes,fontweight='bold',fontsize=15,va='bottom',ha='left')

plt.tight_layout(w_pad=2.8); plt.subplots_adjust(bottom=0.21,top=0.89,left=0.10)
for ext,kw in [('pdf',{}),('png',{'dpi':600}),('tiff',{'dpi':600,'pil_kwargs':{'compression':'tiff_lzw'}})]:
    plt.savefig(os.path.join(OUT_DIR,f'Figure1_calibration.{ext}'),bbox_inches='tight',facecolor='white',**kw)
print(f"Figure 1 saved (bars {fp}; observed r={r_obs:.3f} at {pct}{suffix} percentile)")
