# Equalized Odds

## One-sentence definition

Equalized Odds is a fairness metric for binary classification that requires equal true positive rates and false positive rates across protected groups.

---

## Why it matters

A model can appear highly accurate overall while still making systematically different mistakes for different demographic groups.

Equalized Odds helps detect whether one group is unfairly disadvantaged through:

- more false rejections
- more false approvals
- uneven error behavior across protected attributes

This matters in high-impact decision systems such as:

- hiring pipelines
- loan approval
- insurance risk scoring
- healthcare triage
- criminal justice assessments

If fairness is not measured, bias can remain hidden behind aggregate accuracy metrics.

---

## Real-world example

Suppose a hiring classifier predicts whether candidates should advance to interview rounds.

Results:

| Group | True Positive Rate | False Positive Rate |
|------|-------------------:|-------------------:|
| Group A | 0.91 | 0.07 |
| Group B | 0.69 | 0.18 |

Interpretation:

- qualified candidates in Group B are selected less often
- unqualified candidates in Group B are incorrectly selected more often

Even if total model accuracy looks acceptable, this indicates unequal treatment.

This violates Equalized Odds because both TPR and FPR differ significantly.

---

## How to detect it in Python

```python
from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate
import pandas as pd

# example data
y_true = [1, 1, 1, 0, 0, 0, 1, 0]
y_pred = [1, 1, 0, 1, 0, 0, 1, 0]
group = ["A", "A", "B", "A", "B", "B", "A", "B"]

metrics = {
    "TPR": true_positive_rate,
    "FPR": false_positive_rate
}

frame = MetricFrame(
    metrics=metrics,
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=group
)

print(frame.by_group)
```

Expected output:

```python
     TPR   FPR
A   1.00  0.50
B   0.00  0.00
```

Large differences between groups indicate Equalized Odds violations.

Install dependency:

```bash
pip install fairlearn
```

---

## Mathematical definition

Equalized Odds requires:

P(Ŷ = 1 | Y = 1, A = a) = P(Ŷ = 1 | Y = 1, A = b)

and

P(Ŷ = 1 | Y = 0, A = a) = P(Ŷ = 1 | Y = 0, A = b)

Where:

- Ŷ = model prediction
- Y = true label
- A = protected attribute

Equivalent interpretation:

- equal True Positive Rate
- equal False Positive Rate

---

## Limitations / trade-offs

Equalized Odds is useful, but not universally sufficient.

### Trade-offs

Optimizing for Equalized Odds may:

- reduce overall model accuracy
- conflict with calibration
- conflict with demographic parity
- require threshold adjustments across groups

Not all fairness definitions can be satisfied simultaneously.

---

## Related concepts

### Equal Opportunity

A relaxed version of Equalized Odds.

Requires only equal true positive rates.

---

### Demographic Parity

Requires equal positive prediction rates regardless of ground truth.

Unlike Equalized Odds, it does not account for correctness.

---

### Calibration

Requires prediction scores to have consistent interpretation across groups.

A calibrated model may still violate Equalized Odds.

---

## Further reading

- Hardt, Price, Srebro (2016): Equality of Opportunity in Supervised Learning
- Fairlearn documentation
- Responsible AI fairness evaluation resources
