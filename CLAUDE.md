# Trump Tweets Classification Project - CLAUDE.md

## Project Overview
This is an authorship attribution project for classifying Donald Trump's tweets to determine whether they were written by Trump himself (Android device) or his staffers (iPhone device). The project implements 5 different machine learning algorithms as part of Assignment 3 for text classification and authorship attribution.

## Key Information
- **Assignment**: Text classification and Authorship Attribution (Assignment 3)
- **Deadline**: 23:59, Monday, June 9, 2025
- **Task**: Binary classification of Trump's tweets (0=Trump/Android, 1=Staffer/iPhone)
- **Required Algorithms**: 5 total (Logistic Regression, SVM with linear/nonlinear kernels, FFNN with PyTorch, 4th classifier of choice, 5th transformer-based classifier)

## Required Functions API
The submission notebook must implement these functions:
- `training_pipeline(alg, train_fn)`
- `retrain_best_model(train_fn = None)` 
- `predict(m, fn)`
- `who_am_i()`

## Data Files
- **Training data**: `data/trump_train.tsv` (TSV format: tweet_id, user_handle, tweet_text, timestamp, device)
- **Test data**: Will be provided separately (200 tweets, lacks tweet_id and device fields)
- **Submission format**: Single space-separated line of 0s and 1s in results file

## Libraries & Environment
- **Environment**: Google Colab (must be self-contained)
- **Core libraries**: sklearn, pytorch, nltk, pandas, numpy
- **Special**: Huggingface transformers for 5th classifier
- **Requirements file**: `requirements.txt` available

## Project Structure
- `src/`: Source code organized by functionality
  - `preprocessing/`: Text cleaning, feature extraction, data loading
  - `models/`: ML model implementations (traditional_ml.py, neural_networks.py, etc.)
  - `evaluation/`: Metrics, cross-validation, visualizations
  - `utils/`: Configuration and helper functions
- `models/`: Saved trained models (logistic_regression/, svm/, ffnn/, etc.)
- `data/`: Raw and processed datasets
- `experiments/`: Development notebooks and analysis
- `results/`: Model predictions and performance metrics  
- `notebook/`: Main submission notebook (`submission_notebook.ipynb`)
- `report/`: 3-page PDF report and figures
- `plan/`: Implementation planning documents

## Key Files to Work With
- **Main notebook**: `notebook/submission_notebook.ipynb` 
- **Implementation plan**: `plan/implementation_plan.md`
- **Assignment instructions**: `instructions/Assingement_Instructions.txt`
- **Training data**: `data/trump_train.tsv`
- **Requirements**: `requirements.txt`

## Data Format Details
- **Training data format**: `<tweet_id> <user_handle> <tweet_text> <timestamp> <device>`
- **User handles**: realDonaldTrump, POTUS, PressSec
- **Device types**: android, iphone, instagram, others
- **Timestamp format**: '%Y-%m-%d %H:%M:%S'
- **Labels**: 0 = Trump (Android), 1 = Staffer (iPhone/other)

## Algorithm Requirements
1. **sklearn.linear_model.LogisticRegression**
2. **sklearn.svm.SVC** (both linear and nonlinear kernels)
3. **PyTorch FFNN** (at least one hidden layer)
4. **Fourth classifier** (neural or traditional, allows combining features)
5. **Transformer-based** (MLM or LLM, e.g., BERT/RoBERTa)

## Cross-Validation & Evaluation
- Use sklearn's cross-validation module
- Evaluation metrics: accuracy, precision, recall, F1-score
- Think carefully about evaluation measures
- Report results in table format

## Submission Requirements
1. **Report**: 3-page PDF (`<id>.pdf`) - 11pt font, 1.5 line spacing
   - 1.5 pages: settings and results
   - 1.5 pages: analysis and insights
2. **Results file**: `<id>_aa.txt` - space-separated 0s and 1s
3. **Notebook**: `ex3_<id>.ipynb` - with required API functions
4. **Archive**: `<id>.tar.gz` containing all files

## Report Must Include
- Data preprocessing steps and justification
- Features used for each algorithm
- Data representation for each algorithm
- Hyperparameters and settings
- Algorithm comparison and performance differences
- Best model specification and parameters

## Important Notes
- Code must run in reasonable time (not hours)
- Focus on model interpretability and analysis
- Document all decisions and assumptions
- Test notebook in clean Colab environment before submission

## User Guidelines for Working with Claude

### When Starting Work
- Always reference this CLAUDE.md file to understand project context
- Check the implementation plan in `plan/implementation_plan.md` for current phase
- Review assignment instructions in `instructions/Assingement_Instructions.txt` if needed

### Code Development Approach
- Follow the 8-phase implementation plan in `plan/implementation_plan.md`
- Work incrementally: complete one algorithm before moving to the next
- Test each component thoroughly before integration
- Maintain clean, documented code throughout development

### File Organization Guidelines
- Save preprocessing outputs to `data/processed/` for reuse
- Store trained models in appropriate `models/` subdirectories
- Keep experimental work in `experiments/` notebooks
- Save results and metrics to `results/` directory structure

### Critical Reminders
- Must implement ALL 4 required API functions in submission notebook
- SVM must include BOTH linear and nonlinear kernels
- FFNN must have at least one hidden layer using PyTorch
- Transformer model is required as the 5th classifier
- All code must run in Google Colab environment
- Final predictions must be single line of space-separated 0s and 1s

### Best Practices
- Always use cross-validation for model evaluation
- Document preprocessing decisions and hyperparameter choices
- Compare model performance systematically
- Keep track of best performing configurations
- Test final notebook end-to-end before submission