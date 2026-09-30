# Los Angeles Crime Hotspot Analysis & Incident Classification

An end-to-end geospatial analysis and machine learning pipeline analyzing ~900,000 LAPD crime records. This project identifies high-density crime corridors using spatial clustering and predicts violent vs. property crime incidents using gradient-boosted trees optimized for high recall on rare, high-severity events.

---

## 📌 Executive Summary & Key Results

* **Data Volume:** 899,000+ LAPD crime incident records.
* **Geospatial Hotspots:** Spatially partitioned incident density using coordinate clustering across the Greater Los Angeles area.
* **Recall-Optimized Classification:** Addressed severe class imbalance where baseline models missed **96.3% of violent crimes** (achieving only **3.7% baseline recall**).
* **Calibrated Performance:** Calibrated decision thresholds via Precision-Recall analysis and cost-sensitive weighting, increasing **violent crime detection recall to 70.0%** (identifying 37,777 true violent incidents on the test set).

---

## ⚖️ The Core ML Challenge: Severe Class Imbalance

In municipal crime datasets, violent incidents represent a minority class (~30%) compared to property and municipal infractions.

### Why Standard Classifiers Fail
A standard classifier optimizing for overall accuracy achieves ~70% accuracy simply by predicting "Non-Violent" across the board, resulting in an unacceptable operational failure rate (**3.7% recall** on violent offenses).

### Engineering Solution: Cost-Sensitive Calibration
To align the model with real-world public safety operations—where missing an active violent threat carries far higher costs than a false alarm:
1. **Balanced Class Weights:** Adjusted gradient penalty weights inversely proportional to class frequencies inside `HistGradientBoostingClassifier`.
2. **Decision Threshold Calibration:** Tuned the decision threshold using Precision-Recall curve analysis to hit a minimum **70% recall target** on violent crimes.

### Quantitative Comparison

| Metric | Baseline Classifier | Recall-Tuned Classifier | Operational Impact |
| :--- | :---: | :---: | :--- |
| **Violent Crime Recall** | **3.7%** | **70.0%** | **+66.3% increase in threats flagged** |
| **Violent Incidents Caught** | ~1,996 | **37,777** | **35,781 additional violent crimes captured** |
| **False Negatives (Missed)** | ~51,970 | **16,189** | Massive reduction in undetected violent events |
| **Overall Macro F1** | 0.42 | **0.55** | Balanced performance across both classes |

---

## 📊 Evaluation & Confusion Matrix

### Tuned Confusion Matrix (Test Set: 179,819 Samples)

```text
                  Predicted Non-Violent    Predicted Violent
Actual Non-Violent        63,160                 62,693
Actual Violent            16,189                 37,777  <-- (70% True Positives Detected)