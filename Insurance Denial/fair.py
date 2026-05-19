import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================
# INSURANCE DENIAL BIAS AUDIT — FAIR MODEL
# Dataset: Insurance Claim Analysis: Demographic & Health
# https://www.kaggle.com/datasets/thedevastator/insurance-claim-analysis-demographic-and-health
#
# Protected attributes removed: age, sex
# Proxy variables removed:      bmi, smoker
#
# Retained: only features a person can directly control or
# that reflect the policy itself — not who the person is.
# ============================================================

df = pd.read_csv('insurance.csv')

df['age_group'] = df['age'].apply(lambda x: 'Young (<35)' if x < 35 else 'Older (35+)')

# ── THE FIX: Policy signals only ────────────────────────────
# We keep only features that describe the insurance policy
# context and documented medical history — not demographic
# characteristics that could encode protected class signal.
X = pd.get_dummies(df[[
    'children',     # number of dependants — a policy-level fact
    'region',       # geographic region — retained as a policy factor
    # age     removed ✓  (protected attribute)
    # sex     removed ✓  (protected attribute)
    # bmi     removed ✓  (proxy: encodes race via population distributions)
    # smoker  removed ✓  (proxy: encodes income/class → race)
]])

y = df['insuranceclaim']

# ── TRAIN FAIR MODEL ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# ── MEASURE FAIRNESS GAP ─────────────────────────────────────
df_test = X_test.copy()
df_test['age_group'] = df.loc[X_test.index, 'age_group'].values
df_test['sex'] = df.loc[X_test.index, 'sex'].values
df_test['prediction'] = model.predict(X_test)

age_approval = df_test.groupby('age_group')['prediction'].mean()
sex_approval = df_test.groupby('sex')['prediction'].mean()

print("=" * 60)
print("FAIR MODEL — RESULTS")
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
print("WHAT CHANGED")
print("=" * 60)
print("""
THE FIX: Drop the protected attributes AND their proxies.

  age     → removed. Age is a protected characteristic.
             Young patients were denied at higher rates
             not because of medical risk, but because
             age itself was a training signal.

  sex     → removed. Gender discrimination in insurance
             is illegal under the ACA. Removing it
             eliminates the channel through which the
             model learned to penalise women.

  bmi     → removed. BMI is not an independent health
             signal — it is partially a function of race,
             ethnicity, and socioeconomic status. A model
             that penalises high BMI is partially
             penalising race, regardless of whether
             "race" appears anywhere in the feature list.

  smoker  → removed. Smoking rates correlate with income
             and education. Including smoker status allows
             the model to encode class (and by extension
             racial) signal through an apparently neutral
             variable.

Key Insight: Insurance AI models don't need to name race
to discriminate by race. BMI and smoking are the custody
status of health insurance — legitimate-seeming features
that carry protected-class signal because of structural
inequalities in American society.
""")