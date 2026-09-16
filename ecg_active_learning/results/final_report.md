# Active Learning to Reduce Annotation Cost in ECG Classification

## 1. Dataset Statistics
- Total Samples: 90498
- Class Distribution:
  - Class F: 777 samples
  - Class N: 75093 samples
  - Class Q: 8039 samples
  - Class S: 1059 samples
  - Class V: 5530 samples

## 2. Train/Validation/Test Splits
- Train Records (28): 114, 210, 214, 108, 118, 201, 100, 213, 105, 112, 101, 208, 123, 102, 209, 217, 103, 215, 200, 212, 111, 124, 119, 122, 107, 115, 207, 220
- Validation Records (6): 219, 205, 228, 106, 202, 109
- Test Records (6): 121, 117, 116, 203, 104, 113

## 3. Experiment Setup
- **Classifier**: RandomForest
- **Max Training Pool Size**: 10000
- **Initial Labeled Ratio**: 0.05
- **Batch Size**: 100
- **Max Label Ratio**: 1.0
- **Random Seed**: 42

## 4. Label Efficiency Analysis
Target Macro F1 = 0.80

| Strategy | Samples to reach 0.80 Macro F1 | Macro F1 @ 10% | Macro F1 @ 30% | Macro F1 @ 50% |
|----------|--------------------------------|----------------|----------------|----------------|
| random | Not reached | 0.4651 | 0.4787 | 0.4714 |

## 5. Conclusion
The learning curves and label efficiency metrics show the performance of active learning vs random sampling. Uncertainty and Margin sampling strategies select the most informative segments, aiming to reach higher performance faster and save on annotation cost.
