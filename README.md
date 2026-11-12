# 🚢 PortGuard AI
### AI-Powered Port Congestion and Shipment Delay Prediction System

## 1. Project Overview

PortGuard AI is a machine-learning-based application designed to predict the likelihood of port congestion using operational and environmental factors.

The system uses a **Random Forest Classifier** to analyse port conditions and classify them into two categories:

- **Normal:** Conditions are not classified as congested.
- **Congestion:** Conditions indicate a likelihood of port congestion.

The application also provides an interactive Streamlit dashboard that enables users to explore dataset insights, generate predictions, view congestion probabilities, and understand individual input sensitivities.

**Dataset note:** This project uses a synthetically generated dataset for development, experimentation, and demonstration. It does not currently use live port operational data.

## 2. Project Objectives

- Generate a dataset representing port operating conditions.
- Analyse operational and environmental factors associated with congestion.
- Develop and evaluate a machine-learning classification model.
- Predict congestion using user-provided port conditions.
- Display congestion probability estimates.
- Explain individual input sensitivities using what-if comparisons.
- Present results through an interactive dashboard.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine-learning pipeline and evaluation |
| Random Forest | Classification algorithm |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Joblib | Model serialization |
| Streamlit | Interactive web dashboard |

## 4. Dataset Description

The project uses **500 synthetically generated records** representing five Indian ports:

- Chennai
- Mumbai
- Visakhapatnam
- Kolkata
- Cochin

### Dataset Features

| Feature | Description |
|---|---|
| `port_name` | Name of the port |
| `vessel_arrivals` | Number of vessel arrivals |
| `port_capacity` | Port handling capacity |
| `queue_length` | Number of vessels waiting |
| `weather_severity` | Weather severity score |
| `historical_avg_delay` | Historical average delay |
| `hour_of_day` | Hour of observation |
| `month` | Month of observation |
| `delay_hours` | Simulated delay in hours |
| `congestion` | Target classification |

### Target Variable

The target variable is defined as:

- `0` – No Congestion
- `1` – Congestion

A record is labelled as congested when its simulated delay is **greater than or equal to 6 hours**. This threshold is an assumption used for this prototype.

### Dataset Distribution

| Category | Records | Percentage |
|---|---:|---:|
| Congested | 356 | 71.2% |
| Normal | 144 | 28.8% |
| **Total** | **500** | **100%** |

The synthetic delay values are generated using operational and environmental factors, including vessel arrivals, port capacity, queue length, weather severity, historical delay, time-of-day effects, seasonal effects, and random variation.

## 5. Machine Learning Approach

The project uses a **Random Forest Classifier** to predict port congestion.

### Data Preprocessing

- One-hot encoding is applied to the categorical `port_name` feature.
- Numerical features are passed through the preprocessing pipeline.
- The dataset is divided into training and testing sets.
- Stratified splitting is used to preserve class proportions.
- The target-derived `delay_hours` column is excluded from model inputs to prevent target leakage.

### Training Configuration

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Number of estimators | 200 |
| Maximum depth | 10 |
| Minimum samples split | 5 |
| Minimum samples leaf | 2 |
| Class weighting | Balanced |
| Random state | 42 |
| Training records | 400 |
| Testing records | 100 |

The preprocessing and classification stages are combined into a Scikit-learn pipeline.

## 6. Model Evaluation

The model was evaluated using a held-out test dataset containing 100 records.

### Evaluation Metrics

| Metric | Result |
|---|---:|
| Accuracy | 87.00% |
| Precision (Congestion class) | 90.28% |
| Recall (Congestion class) | 91.55% |
| F1-score (Congestion class) | 90.91% |
| ROC-AUC | 92.67% |

### Confusion Matrix

```text
                 Predicted
                 Normal  Congested

Actual Normal      22        7
Actual Congested    6       65
```

### Interpretation

- 22 normal cases were correctly classified.
- 65 congested cases were correctly identified.
- 7 normal cases were incorrectly classified as congested.
- 6 congested cases were incorrectly classified as normal.

According to the model's global feature-importance analysis, the most influential features were:

1. Queue length – 33.99%
2. Vessel arrivals – 22.19%
3. Historical average delay – 14.28%
4. Weather severity – 10.52%

These metrics and feature-importance values describe the synthetic-data experiment only. They should not be interpreted as verified real-world port forecasting performance.

## 7. Explainable Prediction

PortGuard AI includes an individual feature sensitivity analysis to make model predictions easier to interpret.

### How It Works

1. A baseline input is created using typical values from the dataset.
2. One input feature at a time is replaced with the corresponding user-provided value.
3. The model calculates a new congestion probability.
4. The difference between the new probability and baseline probability is displayed.
5. The dashboard highlights the three largest individual input sensitivities.

### Example Prediction

For a high-congestion Chennai scenario:

| Input | Value |
|---|---:|
| Vessel arrivals | 42 |
| Port capacity | 30 |
| Queue length | 18 |
| Weather severity | 7 |
| Historical average delay | 6.5 hours |
| Hour of day | 18 |
| Month | 8 |

**Model output:** Congestion Detected

**Estimated congestion probability:** 93.96%

The sensitivity analysis identified vessel arrivals, weather severity, and queue length as the three largest positive individual changes relative to the baseline in this scenario.

**Important:** This is a one-feature-at-a-time sensitivity analysis. It does not establish causality and is not an exact SHAP explanation. Individual changes are not additive, and feature interactions can affect model predictions.

## 8. Project Structure

```text
PortGuard-AI/
│
├── data/
│   └── port_congestion_data.csv
│
├── models/
│   └── congestion_model.pkl
│
├── reports/
│   ├── congestion_distribution.png
│   ├── delay_by_port.png
│   ├── arrivals_vs_delay.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── model_evaluation.txt
│
├── generate_data.py
├── analyze_data.py
├── train_model.py
├── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 9. Installation and Setup

### Prerequisites

- Python 3.10 or later recommended
- Git
- pip

### Step 1: Clone the repository

```bash
git clone https://github.com/sanjayneela1929/PortGuard-AI.git
```

### Step 2: Navigate into the project

```bash
cd PortGuard-AI
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the environment

**Windows:**

```bash
.venv\Scripts\activate
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Generate the synthetic dataset

```bash
python generate_data.py
```

### Step 7: Perform exploratory data analysis

```bash
python analyze_data.py
```

### Step 8: Train the machine-learning model

```bash
python train_model.py
```

### Step 9: Run the prediction script (optional)

```bash
python predict.py
```

### Step 10: Launch the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

**Note:** The dataset and trained model must exist before launching the dashboard. The dataset generation and model training steps create these files.

## 10. Application Features

- Synthetic port congestion dataset generation.
- Dataset overview and summary metrics.
- Congestion distribution visualization.
- Average delay comparison across ports.
- Interactive user-input-based congestion prediction.
- Congestion probability display.
- Individual feature sensitivity analysis.
- Baseline probability comparison.
- Top three input sensitivity explanations.
- Dataset preview.
- Standalone command-line prediction script.

## 11. Limitations and Future Enhancements

### Current Limitations

- The dataset is synthetically generated.
- The model has not been validated using real port operational data.
- Predictions depend on the quality and ranges of user-provided inputs.
- Predicted probabilities have not been independently calibrated.
- The sensitivity analysis is not a causal or exact feature-attribution method.

### Future Enhancements

- Integrate real-time port and vessel tracking data.
- Incorporate live weather information.
- Validate the model using real operational records.
- Explore additional machine-learning algorithms.
- Implement more formal local explainability methods.
- Add historical prediction tracking.
- Deploy the application to a cloud platform.

## 12. Conclusion

PortGuard AI demonstrates an end-to-end machine-learning prototype for classifying port congestion conditions using operational and environmental features.

The project combines synthetic data generation, exploratory data analysis, machine-learning model development, evaluation, interactive predictions, and an explainable sensitivity analysis dashboard.

The results demonstrate the feasibility of the proposed approach within the synthetic dataset experiment. Further validation using real-world port operational data would be necessary before considering operational deployment.

---

**Developed by:** N Sanjay Bhargav

**Project:** PortGuard AI – AI-Powered Port Congestion and Shipment Delay Prediction System

**Repository:** https://github.com/sanjayneela1929/PortGuard-AI