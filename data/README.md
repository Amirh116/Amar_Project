# NHANES 2021-2023 Data Source

The raw data files are automatically downloaded from the official CDC/NCHS website via `phase1.py`:
- `DEMO_L.XPT`: Demographic Variables and Sample Weights
- `BMX_L.XPT`: Body Measures (BMI, Waist Circumference)
- `BPXO_L.XPT`: Blood Pressure - Oscillometric (SBP, DBP)

Raw `.XPT` files are excluded from version control due to file size constraints and are reproducible via code.
