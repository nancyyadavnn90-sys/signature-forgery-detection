# Signature Forgery Detection using Machine Learning

Feature engineering, regularization and ensemble-based classification on the **CEDAR** signature database.
CSE2032 Machine Learning, BML Munjal University (AY 2026-27).

| Team member | Roll No. |
|---|---|
| Nancy Yadav | 240710 |
| Purva Joshi | 240727 |
| Jiya Joshi | 240954 |

**Live demo:** <https://signature-forgery-detection-c5xxqjncxxfpevbjz4mwvj.streamlit.app>
(Streamlit Community Cloud) · **Code:** <https://github.com/nancyyadavnn90-sys/signature-forgery-detection>

## What the system does

Given a scanned **questioned signature** and a few **genuine reference signatures** of the same person
(the "specimen signatures on file" that a bank keeps), the system decides whether the questioned
signature is **Genuine** or **Forged**:

1. **Pre-processing:** grayscale, 3x3 median filter, Otsu thresholding, removal of noise specks, cropping.
2. **Feature derivation:** 51 hand-crafted features (stroke thickness & pressure, size & shape including
   aspect ratio, curvature & tremor, 3x4 pixel-density grid, projection profiles).
3. **Writer normalisation:** every feature is expressed as a deviation from the writer's reference
   signatures, plus three distance statistics.
4. **Classification:** Logistic Regression (no penalty / L1 / L2), SVM, KNN, Decision Tree, Random Forest,
   Gradient Boosting, AdaBoost and Soft Voting, tuned with `GridSearchCV` + writer-grouped 5-fold CV and
   evaluated on 11 writers never seen in training (accuracy, precision, recall, F1, ROC-AUC, FAR, FRR, EER).

<!-- RESULTS START -->
## Results (11 unseen test writers, 473 questioned signatures)

Best model by cross-validation: **AdaBoost** - accuracy **90.3%**, ROC-AUC **0.970**,
EER **9.9%**, FAR 10.2%, FRR 9.1% (threshold 0.5).
Best on the test writers: **Random Forest** (ROC-AUC 0.983).
The top cross-validation scores differ by less than one standard deviation, so the CV winner is partly
luck (winner's curse); the model was not switched after looking at the test set.

| Model | CV ROC-AUC | Test accuracy | F1 | ROC-AUC | FAR | FRR | EER |
|---|---|---|---|---|---|---|---|
| AdaBoost | 0.978 | 90.3% | 0.912 | 0.970 | 10.2% | 9.1% | 9.9% |
| Soft Voting | 0.976 | 93.2% | 0.939 | 0.982 | 6.4% | 7.2% | 6.8% |
| Gradient Boosting | 0.974 | 92.4% | 0.932 | 0.973 | 6.4% | 9.1% | 7.5% |
| Random Forest | 0.973 | 92.2% | 0.931 | 0.983 | 4.9% | 11.5% | 6.1% |
| LogReg + L1 (Lasso) | 0.972 | 90.1% | 0.910 | 0.958 | 9.5% | 10.5% | 10.8% |
| SVM | 0.972 | 92.6% | 0.933 | 0.981 | 7.2% | 7.7% | 7.4% |
| LogReg + L2 (Ridge) | 0.971 | 91.1% | 0.920 | 0.965 | 8.0% | 10.0% | 9.3% |
| LogReg (no penalty) | 0.970 | 89.0% | 0.901 | 0.949 | 10.6% | 11.5% | 11.0% |
| KNN | 0.966 | 85.4% | 0.852 | 0.974 | 24.6% | 1.9% | 7.2% |
| Decision Tree | 0.958 | 91.1% | 0.923 | 0.945 | 5.3% | 13.4% | 9.6% |

Operating points of AdaBoost (thresholds chosen on training-CV predictions only):

| Operating point | Threshold | Accuracy | FAR | FRR |
|---|---|---|---|---|
| Default | 0.500 | 90.3% | 10.2% | 9.1% |
| EER (FAR = FRR on train CV) | 0.512 | 89.9% | 11.4% | 8.6% |
| High security (train-CV FAR <= 5%) | 0.456 | 92.0% | 4.9% | 12.0% |

Absolute features (Experiment 1, best model Gradient Boosting): accuracy 74.2%,
ROC-AUC 0.831, EER 24.3% -> writer-normalised features (Experiment 2):
accuracy 90.3%, ROC-AUC 0.970, EER 9.9%.
<!-- RESULTS END -->

## Project structure

```
ML_Project/
├── notebooks/Signature_Forgery_Detection.ipynb   <- complete pipeline with outputs (main code)
├── src/                                          <- reusable modules used by the notebook and the app
│   ├── config.py        paths, seeds, settings
│   ├── data.py          dataset indexing and image loading
│   ├── features.py      pre-processing + 51 hand-crafted features
│   ├── reference.py     writer-normalised (reference-based) features
│   ├── modeling.py      models, hyper-parameter grids, tuning loop
│   ├── evaluation.py    accuracy ... ROC-AUC, FAR, FRR, EER, threshold selection
│   ├── analysis.py      EDA statistics, correlation filter
│   ├── plotting.py      chart style
│   └── verify.py        inference for the demo app
├── app/streamlit_app.py  <- live demo (deployable on Streamlit Community Cloud)
├── app/samples/          <- signatures of 4 unseen test writers for the demo
├── models/               <- trained model (signature_verifier.joblib) + model_card.json
├── outputs/figures/      <- all figures used in the report and slides
├── outputs/tables/       <- all result tables (CSV)
├── outputs/features_raw.csv  <- extracted features of all 2,640 images (cache)
├── report/               <- project report (DOCX + PDF; submitted separately)
├── presentation/         <- slides (PPTX + PDF; submitted separately)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
pip install notebook ipykernel          # to open / re-run the notebook
```

### Dataset

CEDAR signature database (Center of Excellence for Document Analysis and Recognition, University at Buffalo):
55 writers x (24 genuine + 24 forged) = 2,640 images.

* Original: <https://cedar.buffalo.edu/NIJ/data/signatures.rar>
* Kaggle mirror: <https://www.kaggle.com/datasets/shreelakshmigp/cedardataset>

Extract it so that the folders are `data/CEDAR/full_org/` and `data/CEDAR/full_forg/`.
(The extracted features are already cached in `outputs/features_raw.csv`; the images are only needed to
re-extract features and to draw the image figures.)

## Run

```bash
# full pipeline (EDA -> features -> models -> evaluation -> saves the model)
jupyter notebook notebooks/Signature_Forgery_Detection.ipynb

# live demo
streamlit run app/streamlit_app.py
```

## Deploy (Streamlit Community Cloud, free)

The app is deployed at <https://signature-forgery-detection-c5xxqjncxxfpevbjz4mwvj.streamlit.app>.
To deploy your own copy:

1. Push this folder to a public GitHub repository (the `data/` folder is not needed; keep `models/`,
   `app/`, `src/` and `requirements.txt`).
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**.
3. Choose the repository, branch `main`, main file path `app/streamlit_app.py`; under *Advanced settings*
   select **Python 3.11**. Click **Deploy**.
4. The app gets a public URL like `https://<your-app>.streamlit.app`.
