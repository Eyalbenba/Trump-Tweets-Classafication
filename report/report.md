# Trump Tweets Classification Report

## Introduction

This report presents an authorship attribution analysis of Donald Trump's tweets, distinguishing between tweets written by Trump himself (using Android devices) and those written by his staffers (using iPhones). The analysis employs five different machine learning algorithms to perform binary classification.

## Project Structure

The project is organized into a comprehensive directory structure to support systematic development and analysis:

```
Trump-Tweets-Classification/
├── data/                    # Raw and processed datasets
├── src/                     # Source code organized by functionality
│   ├── preprocessing/       # Data preprocessing modules
│   ├── models/             # Model implementations
│   ├── evaluation/         # Evaluation and metrics
│   └── utils/              # Utility functions
├── models/                  # Saved trained models
├── experiments/            # Experimental notebooks and analysis
├── results/                # Results and predictions
├── notebook/               # Main submission notebook
├── report/                 # Report documentation
└── output/                 # Final submission files
```

This structure enables modular development, reproducible experiments, and organized documentation throughout the project lifecycle.

## Data Description

- **Dataset**: Trump tweets from early 2015 to mid 2017
- **Format**: Tab-separated values with fields: tweet_id, handle, text, timestamp, device
- **Labels**: 
  - 0: Trump (Android device)
  - 1: Staffer (iPhone device)
- **Task**: Binary classification for authorship attribution

## Methodology

### Data Preprocessing
[To be filled with preprocessing steps]

### Algorithms Implemented

1. **Logistic Regression** (sklearn.linear_model.LogisticRegression)
2. **Support Vector Machine** (sklearn.svm.SVC) - both linear and nonlinear kernels
3. **Feed-Forward Neural Network** (PyTorch) - with at least one hidden layer
4. **[Fourth Classifier Name]** - [Description]
5. **[Fifth Classifier Name]** - Transformer-based approach

### Feature Engineering
[To be filled with feature extraction details]

### Model Training and Evaluation
[To be filled with training details and cross-validation approach]

## Results

### Performance Comparison
[Table with results for each algorithm]

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| Logistic Regression | - | - | - | - |
| SVM (Linear) | - | - | - | - |
| SVM (RBF) | - | - | - | - |
| FFNN | - | - | - | - |
| [Fourth Classifier] | - | - | - | - |
| [Fifth Classifier] | - | - | - | - |

## Analysis and Insights

### Preprocessing Justification
[Analysis of preprocessing choices and their impact]

### Feature Representation
[Discussion of data representation for each algorithm]

### Hyperparameter Settings
[Details of optimal parameters for each model]

### Algorithm Comparison
[Analysis of performance differences between algorithms]

### Best Performing Model
[Specification of the best model and parameters used for final predictions]

## Conclusions

[Summary of findings and insights from the authorship attribution task]

## References

[Any relevant references used in the analysis]