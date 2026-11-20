import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# ایجاد پوشه ذخیره نمودارها
os.makedirs('reports/figures', exist_ok=True)

# ۱. بارگذاری داده‌ها
df_demo = pd.read_sas('data/DEMO_L.XPT', format='xport')
df_bmx = pd.read_sas('data/BMX_L.XPT', format='xport')
df_bp = pd.read_sas('data/BPXO_L.XPT', format='xport')

# ادغام شامل ستون وزن (WTINT2YR)
df = df_demo.merge(df_bmx, on='SEQN').merge(df_bp, on='SEQN')

# ۲. پیش‌پردازش
df['SBP'] = df[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].mean(axis=1)
# حفظ ستون وزن برای محاسبات بعدی
df = df[['RIAGENDR', 'RIDAGEYR', 'BMXBMI', 'SBP', 'WTINT2YR']].copy()
df.columns = ['Gender', 'Age', 'BMI', 'SBP', 'Weight']
df = df[df['Age'] >= 20].dropna()
df['Gender'] = df['Gender'].replace({1: 'Male', 2: 'Female'})

df['High_BP'] = df['SBP'] > 130
df['Old'] = df['Age'] > 40

# ۳. محاسبه نسبت بدون وزن در مقابل وزن‌دار
unweighted_prop = df['High_BP'].mean()
weighted_prop = (df['High_BP'] * df['Weight']).sum() / df['Weight'].sum()

print("=" * 50)
print(f"Comparison of High_BP Proportion (Weighted vs Unweighted):")
print(f"Unweighted Proportion: {unweighted_prop:.4f}")
print(f"Weighted Proportion  : {weighted_prop:.4f}")
print(f"Difference           : {abs(unweighted_prop - weighted_prop):.4f}")
print("=" * 50)

# ۴. بخش احتمالات قبلی
P_HighBP = unweighted_prop # از نسبت اصلی استفاده می‌کنیم
P_Old = df['Old'].mean()
P_Joint = ((df['High_BP']) & (df['Old'])).mean()

print(f"P(High_BP)              : {P_HighBP:.4f}")
print(f"P(Age > 40)             : {P_Old:.4f}")
print(f"P(High_BP and Age > 40) : {P_Joint:.4f}")

# ۵. بررسی استقلال و بیز (همانند قبل)
P_Indep_Check = P_HighBP * P_Old
print(f"Independent? Difference : {abs(P_Joint - P_Indep_Check):.4f}")

# ۶. ترسیم و ذخیره نمودارها
plt.figure(figsize=(8, 5))
count, bins, _ = plt.hist(df['SBP'], bins=35, density=True, alpha=0.6, color='skyblue', edgecolor='black')
mu, std = df['SBP'].mean(), df['SBP'].std()
normal_curve = stats.norm.pdf(bins, mu, std)
plt.plot(bins, normal_curve, 'r-', linewidth=2, label='Normal Fit')
plt.title('Distribution of SBP with Normal Approximation')
plt.savefig('reports/figures/sbp_distribution_fit.png', dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(df['Age'], df['SBP'], alpha=0.2, color='navy', s=10)
plt.title('Scatter Plot of Age vs SBP')
plt.savefig('reports/figures/age_vs_sbp_scatter.png', dpi=300)
plt.close()

print("\nAll calculations complete. Figures saved in 'reports/figures/'.")
