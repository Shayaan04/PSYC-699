"""
IS PROTECTION FOR SALE IN HIGH INCOME COUNTRIES?
Trade Association Density and Non-Tariff Measure Proliferation
Kansas Data Science Consortium / World Bank
Author: Shayaan Mohammed, University of Kansas
"""

from pathlib import Path

OUTPUT_DIR = Path(r"C:\Users\shaya\OneDrive\Desktop\699report")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = OUTPUT_DIR / "panel_data_final.csv"

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.tools import add_constant
from linearmodels.panel import RandomEffects
warnings.filterwarnings('ignore')

print("Data file exists    :", os.path.exists(DATA_PATH))
print("Output folder exists:", os.path.exists(OUTPUT_DIR))

plt.rcParams.update({
    'figure.dpi': 150, 'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
})

COLORS = {
    'H': '#2166AC', 'UM': '#91BFDB', 'LM': '#FC8D59',
    'L': '#D73027', 'main': '#2166AC', 'grey': '#636363',
}
INC_ORDER  = ['L', 'LM', 'UM', 'H']
INC_LABELS = {'L':'Low','LM':'Lower-Middle','UM':'Upper-Middle','H':'High'}

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("\nLoading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df['year']      = pd.to_numeric(df['year'],      errors='coerce')
df['isic_rev4'] = pd.to_numeric(df['isic_rev4'], errors='coerce')
df = df[df['year'] >= 1991].copy()
df = df[df['ISO_A3'] != 'EU'].copy()

df['ln_new_ntm']     = np.log1p(df['new_ntm_count'])
df['ln_cum_ntm']     = np.log1p(df['cumulative_ntm_count'])
df['ln_ta']          = np.log1p(df['trade_association_count'])
df['polity2_clean']  = df['polity2'].where(df['polity2'].between(-10, 10))
df['regime'] = np.where(df['polity2_clean'] > 0, 'Democracy',
               np.where(df['polity2_clean'] <= 0, 'Autocracy', None))

SYNTH = ['emp_synth','va_synth','wages_synth','output_synth','estab_synth']
df['has_controls'] = df[SYNTH].notna().all(axis=1)

# ── REGRESSION DATASET ────────────────────────────────────────────────────────
reg_df = df[df['has_controls']].copy()
reg_df['ln_emp_s']    = np.log1p(reg_df['emp_synth'])
reg_df['ln_va_s']     = np.log1p(reg_df['va_synth'])
reg_df['ln_wages_s']  = np.log1p(reg_df['wages_synth'])
reg_df['ln_output_s'] = np.log1p(reg_df['output_synth'])
reg_df['ln_estab_s']  = np.log1p(reg_df['estab_synth'])
CTRL     = ['ln_emp_s','ln_va_s','ln_wages_s','ln_output_s','ln_estab_s']
ctrl_str = ' + '.join(CTRL)
reg_df = reg_df.dropna(
    subset=['ln_new_ntm','ln_ta','ISO_A3','isic_rev4','year']+CTRL
).reset_index(drop=True)
reg_df['country_fe'] = reg_df['ISO_A3'].astype('category').cat.codes
reg_df['year_fe']    = reg_df['year'].astype('category').cat.codes
fe_str = ' + C(country_fe) + C(year_fe)'

hi_df = reg_df[reg_df['income_group']=='H'].copy().reset_index(drop=True)
hi_df['country_fe'] = hi_df['ISO_A3'].astype('category').cat.codes
hi_df['year_fe']    = hi_df['year'].astype('category').cat.codes

print(f"Full sample:   {len(reg_df):,} obs | {reg_df['ISO_A3'].nunique()} countries")
print(f"High income:   {len(hi_df):,} obs | {hi_df['ISO_A3'].nunique()} countries")

def fit(formula, data):
    return smf.ols(formula, data=data).fit(
        cov_type='cluster', cov_kwds={'groups': data['ISO_A3'].values})

def get_ci(model, var):
    try:
        b  = model.params[var]
        se = model.bse[var] if hasattr(model,'bse') else model.std_errors[var]
        p  = model.pvalues[var]
        return b, b-1.96*se, b+1.96*se, p
    except: return np.nan,np.nan,np.nan,np.nan

def stars(p):
    return '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''

# ── VISUALIZATIONS ────────────────────────────────────────────────────────────
print("\nGenerating visualizations...")
cmap10 = plt.cm.get_cmap('tab10', 10)

# Fig 1 — NTMs by income group over time
fig, ax = plt.subplots(figsize=(12,6))
for grp in INC_ORDER:
    sub = df[df['income_group']==grp].groupby('year')['new_ntm_count'].mean().reset_index()
    ax.plot(sub['year'],sub['new_ntm_count'],color=COLORS[grp],linewidth=2.5,
            label=INC_LABELS[grp],marker='o',markersize=4)
ax.set_title('Average New NTMs Per Year by Income Group (1991–2024)',fontweight='bold')
ax.set_xlabel('Year'); ax.set_ylabel('Avg. New NTMs per Country-Industry')
ax.legend(title='Income Group'); ax.set_xlim(1991,2024)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig1_ntm_by_income_over_time.png', bbox_inches='tight')
plt.close(); print("  ✓ Fig 1")

# Fig 2 — High income vs others
fig, ax = plt.subplots(figsize=(12,6))
hi_ntm  = df[df['income_group']=='H'].groupby('year')['new_ntm_count'].mean()
oth_ntm = df[df['income_group']!='H'].groupby('year')['new_ntm_count'].mean()
ax.plot(hi_ntm.index,hi_ntm.values,color=COLORS['H'],linewidth=2.5,
        label='High Income',marker='o',markersize=4)
ax.plot(oth_ntm.index,oth_ntm.values,color=COLORS['grey'],linewidth=2.5,
        label='All Other Countries',marker='s',markersize=4,linestyle='--')
ax.set_title('Average New NTMs: High Income vs All Other Countries',fontweight='bold')
ax.set_xlabel('Year'); ax.set_ylabel('Avg. New NTMs per Country-Industry')
ax.legend(); ax.set_xlim(1991,2024)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig2_hi_vs_others_ntm.png', bbox_inches='tight')
plt.close(); print("  ✓ Fig 2")

# Fig 3 — Scatter TA vs NTM (high income)
fig, ax = plt.subplots(figsize=(10,7))
hi_scat = df[(df['income_group']=='H')&(df['trade_association_count']>0)
             &(df['new_ntm_count']>0)]
cavg = hi_scat.groupby('ISO_A3').agg(
    avg_ta=('trade_association_count','mean'),
    avg_ntm=('new_ntm_count','mean')).reset_index()
ax.scatter(np.log1p(cavg['avg_ta']),np.log1p(cavg['avg_ntm']),
           color=COLORS['H'],s=100,alpha=0.8,edgecolors='white')
for _,row in cavg.iterrows():
    ax.annotate(row['ISO_A3'],(np.log1p(row['avg_ta']),np.log1p(row['avg_ntm'])),
                fontsize=8,color=COLORS['grey'],xytext=(4,4),textcoords='offset points')
x=np.log1p(cavg['avg_ta']); y=np.log1p(cavg['avg_ntm'])
z=np.polyfit(x,y,1); p2=np.poly1d(z)
xl=np.linspace(x.min(),x.max(),100)
ax.plot(xl,p2(xl),color=COLORS['H'],linewidth=2,linestyle='--',alpha=0.6,label='Trend')
ax.set_title('Trade Association Density vs NTM Intensity\nHigh Income Countries',
             fontweight='bold')
ax.set_xlabel('Log(Avg Trade Associations + 1)')
ax.set_ylabel('Log(Avg New NTMs + 1)')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig3_scatter_hi_ta_ntm.png', bbox_inches='tight')
plt.close(); print("  ✓ Fig 3")

# ── REGRESSIONS ───────────────────────────────────────────────────────────────
print("\nRunning regressions...")

# Model 1 — Pooled OLS
m1 = fit(f'ln_new_ntm ~ ln_ta + {ctrl_str}', hi_df)
b1,lo1,hi1,p1 = get_ci(m1,'ln_ta')
print(f"  ✓ M1 Pooled OLS:    β={b1:.4f}  p={p1:.4f}")

# Model 2 — Random Effects
try:
    re_df2 = hi_df.set_index(['country_fe','year_fe'])
    m2 = RandomEffects(re_df2['ln_new_ntm'],
                       add_constant(re_df2[['ln_ta']+CTRL])).fit(cov_type='robust')
    b2,lo2,hi2,p2 = get_ci(m2,'ln_ta')
    print(f"  ✓ M2 Random Effects: β={b2:.4f}  p={p2:.4f}")
except Exception as e:
    print(f"  ! M2 skipped: {e}")
    m2,b2,lo2,hi2,p2=None,np.nan,np.nan,np.nan,np.nan

# Model 3 — Two-Way Fixed Effects (MAIN MODEL)
m3 = fit(f'ln_new_ntm ~ ln_ta + {ctrl_str}{fe_str}', hi_df)
b3,lo3,hi3,p3 = get_ci(m3,'ln_ta')
print(f"  ✓ M3 Two-Way FE:    β={b3:.4f}  p={p3:.4f}  ← MAIN MODEL")

# Model 4 — FE by income group
m_inc = {}
for grp in INC_ORDER:
    sub = reg_df[reg_df['income_group']==grp].copy().reset_index(drop=True)
    sub['country_fe'] = sub['ISO_A3'].astype('category').cat.codes
    sub['year_fe']    = sub['year'].astype('category').cat.codes
    if len(sub)<100: continue
    try:
        m = fit(f'ln_new_ntm ~ ln_ta + {ctrl_str}{fe_str}',sub)
        m_inc[grp] = m
        b,lo,hi_,p = get_ci(m,'ln_ta')
        print(f"  ✓ M4 {INC_LABELS[grp]:13s}: β={b:.4f}  p={p:.4f}")
    except Exception as e: print(f"  ! M4 {grp}: {e}")

# ── COEFFICIENT PLOTS ─────────────────────────────────────────────────────────
print("\nGenerating coefficient plots...")

def single_coef_plot(coef, lo, hi_, pval, label, title, xlabel, filename,
                     color='#2166AC', marker='o'):
    fig, ax = plt.subplots(figsize=(9,3))
    ax.errorbar(coef, 0, xerr=[[coef-lo],[hi_-coef]],
                fmt=marker, color=color, capsize=8,
                markersize=14, linewidth=3, capthick=2.5)
    ax.text(hi_+abs(hi_-lo)*0.02, 0,
            f' β = {coef:.3f}{stars(pval)}\n p = {pval:.4f}',
            va='center', fontsize=12, color='#1A1A2E')
    ax.axvline(0,color='black',linestyle='--',alpha=0.5,linewidth=1.5)
    ax.set_yticks([0]); ax.set_yticklabels([label],fontsize=12)
    ax.set_xlabel(xlabel,fontsize=11)
    ax.set_title(title,fontweight='bold',fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, bbox_inches='tight')
    plt.close()

# Fig 4 — Pooled OLS
single_coef_plot(b1,lo1,hi1,p1,'Trade Association\nDensity (log)',
    'Model 1: Pooled OLS\nHigh Income Countries',
    'Coefficient on Trade Association Density (log) with 95% CI',
    'fig4_model1_pooled_ols.png',COLORS['main'])
print("  ✓ Fig 4")

# Fig 5 — Random Effects
if m2 is not None:
    single_coef_plot(b2,lo2,hi2,p2,'Trade Association\nDensity (log)',
        'Model 2: Random Effects GLS\nHigh Income Countries',
        'Coefficient on Trade Association Density (log) with 95% CI',
        'fig5_model2_random_effects.png',COLORS['main'])
    print("  ✓ Fig 5")

# Fig 6 — Two-Way FE (MAIN)
single_coef_plot(b3,lo3,hi3,p3,'Trade Association\nDensity (log)',
    'Model 3: Two-Way Fixed Effects (Country + Year)\nHigh Income Countries — Main Model',
    'Coefficient on Trade Association Density (log) with 95% CI',
    'fig6_model3_twoway_fe.png',COLORS['H'])
print("  ✓ Fig 6")

# Fig 7 — Income group comparison
inc_rows = [(INC_LABELS[g],*get_ci(m_inc[g],'ln_ta'),COLORS[g])
            for g in INC_ORDER if g in m_inc]
fig, ax = plt.subplots(figsize=(11,5))
for i,(label,b,lo,hi_,p,color) in enumerate(inc_rows):
    ax.errorbar(b,i,xerr=[[b-lo],[hi_-b]],fmt='o',color=color,
                capsize=6,markersize=12,linewidth=2.5,capthick=2)
    ax.text(hi_+0.003,i,f' {b:.3f}{stars(p)}',va='center',fontsize=10)
ax.axvline(0,color='black',linestyle='--',alpha=0.5,linewidth=1.2)
ax.set_yticks(range(len(inc_rows)))
ax.set_yticklabels([r[0] for r in inc_rows],fontsize=11)
ax.set_xlabel('Coefficient on Trade Association Density (log) with 95% CI',fontsize=11)
ax.set_title('Model 4: Two-Way Fixed Effects by Income Group\n'
             'Is the Effect Strongest in High Income Countries?',fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig7_model4_income_comparison.png', bbox_inches='tight')
plt.close(); print("  ✓ Fig 7")

# ── SAVE RESULTS CSV ──────────────────────────────────────────────────────────
print("\nSaving regression results...")
rows = []
def make_row(label,b,lo,hi_,p,n,r2='N/A'):
    return {'Model':label,'Coefficient':round(b,4),'CI_Lower':round(lo,4),
            'CI_Upper':round(hi_,4),'p_value':round(p,4),'Significance':stars(p),
            'R_squared':r2,'N':n}

rows.append(make_row('M1 Pooled OLS (HI)',b1,lo1,hi1,p1,
                     int(m1.nobs),round(m1.rsquared,4)))
if m2 is not None:
    rows.append(make_row('M2 Random Effects (HI)',b2,lo2,hi2,p2,int(m2.nobs)))
rows.append(make_row('M3 Two-Way FE (HI)',b3,lo3,hi3,p3,
                     int(m3.nobs),round(m3.rsquared,4)))
for grp in INC_ORDER:
    if grp in m_inc:
        b,lo,hi_,p = get_ci(m_inc[grp],'ln_ta')
        rows.append(make_row(f'M4 FE {INC_LABELS[grp]}',b,lo,hi_,p,
                             int(m_inc[grp].nobs),round(m_inc[grp].rsquared,4)))

pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'regression_results.csv', index=False)
print("  ✓ Regression results saved")

print("\n" + "="*60)
print("ALL ANALYSIS COMPLETE")
print("="*60)
print(f"\nFiles saved to: {OUTPUT_DIR}")