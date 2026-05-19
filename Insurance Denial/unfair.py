import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================
# INSURANCE DENIAL BIAS AUDIT — BIASED MODEL
# Dataset: Insurance Claim Analysis: Demographic & Health
# https://www.kaggle.com/datasets/thedevastator/insurance-claim-analysis-demographic-and-health
#
# Protected attributes included: age, sex
# Proxy variables included:      bmi, smoker
#
# BMI is a documented proxy for race — Black and Hispanic
# Americans are flagged as "obese" at higher rates due to
# population-level differences, not individual health risk.
# Smoker status correlates with income and education, which
# themselves correlate with race and class.
# ============================================================

df = pd.read_csv('insurance.csv')

# Define age groups for fairness measurement
df['age_group'] = df['age'].apply(lambda x: 'Young (<35)' if x < 35 else 'Older (35+)')

# ── BIASED FEATURES ─────────────────────────────────────────
# age and sex are protected attributes.
# bmi is a proxy: BMI-based risk scoring disproportionately
#   flags Black and Hispanic patients as high-risk due to
#   population-level BMI distributions, not individual risk.
# smoker is a proxy: smoking rates correlate with poverty and
#   lower education, which are structurally tied to race/class.
X = pd.get_dummies(df[[
    'age',          # protected attribute
    'sex',          # protected attribute
    'bmi',          # proxy: correlated with race via population BMI distributions
    'children',
    'smoker',       # proxy: correlated with income → race/class
    'region',
]])

y = df['insuranceclaim']

# ── PROXY VARIABLE ANALYSIS ──────────────────────────────────
print("=" * 60)
print("PROXY VARIABLE ANALYSIS")
print("=" * 60)
print("\nBMI distribution by age group:")
print(df.groupby('age_group')['bmi'].mean().round(2))

smoker_age = pd.crosstab(df['smoker'], df['age_group'], normalize='columns').round(3)
print("\nSmoker rates by age group:")
print(smoker_age)
print()

# ── TRAIN BIASED MODEL ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# ── MEASURE FAIRNESS GAP ─────────────────────────────────────
df_test = X_test.copy()
df_test['age_group'] = df.loc[X_test.index, 'age_group'].values
df_test['sex_female'] = df_test.get('sex_female', 0)
df_test['prediction'] = model.predict(X_test)

# Re-attach raw sex column for grouping
df_test['sex'] = df.loc[X_test.index, 'sex'].values

# Age fairness gap (claim approval rate = NOT denied)
age_approval = df_test.groupby('age_group')['prediction'].mean()

# Sex fairness gap
sex_approval = df_test.groupby('sex')['prediction'].mean()

print("=" * 60)
print("BIASED MODEL — RESULTS")
print("=" * 60)
print(f"\nModel Accuracy: {accuracy:.2%}\n")

print("── Claim Approval Rate by Age Group ──────────────────")
for group, rate in age_approval.items():
    print(f"  {group:<20} {rate:.2%}")
age_gap = abs(age_approval.iloc[0] - age_approval.iloc[1])
print(f"\n  Fairness Gap (Age):  {age_gap:.2%}")

print("\n── Claim Approval Rate by Sex ────────────────────────")
for group, rate in sex_approval.items():
    print(f"  {group:<20} {rate:.2%}")
sex_gap = abs(sex_approval['male'] - sex_approval['female'])
print(f"\n  Fairness Gap (Sex):  {sex_gap:.2%}")

print("\n" + "=" * 60)
print("WHAT'S WRONG")
print("=" * 60)
print("""
This model includes age and sex as direct inputs — protected
attributes under the ACA and anti-discrimination law.

It also includes two proxy variables:

  BMI      → population-level BMI distributions differ by
              race/ethnicity. Flagging high BMI as a risk
              factor disproportionately penalises Black and
              Hispanic patients independent of actual health
              outcomes.

  Smoker   → smoking rates are inversely correlated with
              income and education. Income and education are
              themselves correlated with race and class.
              'Smoker' smuggles socioeconomic signal —
              and therefore racial signal — back into the
              model even if race is never named.

Run fair.py to see the fix.
""")