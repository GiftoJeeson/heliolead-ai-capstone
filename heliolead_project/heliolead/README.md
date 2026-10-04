# HelioLead AI: Predictive Solar Lead Scoring

End-to-end capstone project: synthetic data generation -> EDA -> feature engineering ->
model training (RandomForest vs XGBoost) -> evaluation -> Streamlit dashboard.

## Project structure
```
heliolead/
├── requirements.txt
├── generate_dataset.py       # Step 1: builds the labeled dataset
├── eda_and_preprocessing.py  # Step 2: EDA + feature engineering + train/test split
├── train_model.py            # Step 3: trains, evaluates, saves the model
├── app.py                    # Step 4: Streamlit lead-scoring dashboard
├── data/                     # generated CSVs (raw + processed splits)
├── figures/                  # saved EDA and evaluation plots
└── models/                   # trained model + metrics.json
```

## 1. Set up the environment in VS Code

1. Open this folder in VS Code (`File > Open Folder`).
2. Install the **Python extension** if you don't have it (Extensions panel, search "Python").
3. Create a virtual environment so this project's packages don't clash with anything else on your machine:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # on Windows: .venv\Scripts\activate
   ```
4. In VS Code, press `Ctrl+Shift+P` -> "Python: Select Interpreter" -> pick the `.venv` one.
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 2. About the dataset (read this before your viva/defense)

There is no public dataset that already pairs "roof orientation + income + bill"
with a real adoption label at the household level — real solar-adoption datasets
(Google Project Sunroof, DeepSolar, US EIA/RECS survey) are aggregated at the
census-tract level and don't give you individual leads to score.

`generate_dataset.py` builds a **synthetic but realistically-grounded** dataset:
feature ranges are based on published solar-irradiance and income-adoption
patterns (cited as comments in the script), and the adoption label is generated
by a logistic (probabilistic) function plus random noise — not a simple rule —
so the classification problem is genuinely learnable but not trivial. This is
the same technique used in industry before a company has enough historical CRM
data to train on real outcomes. Say this explicitly in your report; it's a
strength, not a shortcut, as long as you're transparent about it.

If your evaluators require a real dataset, alternatives to layer in later:
- **NREL NSRDB** (nsrdb.nrel.gov) for real solar irradiance by coordinates
- **data.gov.in** / state discom portals for real electricity tariff slabs
- **Kaggle "Solar Power Generation Data"** for generation (not adoption) data,
  useful for a related but different problem framing

## 3. Run the pipeline, in order

```bash
python generate_dataset.py         # -> data/heliolead_solar_leads.csv
python eda_and_preprocessing.py    # -> figures/*.png, data/X_train.csv etc.
python train_model.py              # -> models/heliolead_model.joblib, metrics.json
streamlit run app.py               # -> opens the dashboard in your browser
```

`eda_and_preprocessing.py` is written with `# %%` cell markers — in VS Code you
can also open it and click "Run Cell" above each block to step through it like
a Jupyter notebook, which is useful for your EDA screenshots in the report.

## 4. What to put in each deliverable

- **Project report**: problem statement (from your brief) + dataset section
  above + EDA figures from `figures/` + `models/metrics.json` for your
  precision/recall/F1 table + a short "limitations" paragraph about synthetic
  data.
- **PPT**: architecture diagram (Data -> Preprocessing -> Model -> Dashboard),
  3-4 key EDA charts, the ROC comparison chart, and screenshots of the running
  Streamlit app.
- **Live demo**: `streamlit run app.py`, score 2-3 example leads live (one
  clearly hot, one clearly cold, one borderline) to show the model isn't just
  predicting one class.

## 5. Natural next steps once the base version works

- Swap RandomForest/XGBoost hyperparameters for a `GridSearchCV` sweep
- Add SHAP values for per-lead explainability in the dashboard ("why is this a hot lead")
- Replace the synthetic generator with real inquiry data once your team has any
- Add a CSV batch-upload mode to `app.py` for scoring a whole lead list at once
