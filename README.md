
# PortGuard AI
### AI-Powered Port Congestion Prediction System

## 1. Project Overview

PortGuard AI is a machine-learning-based application designed to predict port congestion using operational and environmental factors.

The application uses a Random Forest Classifier to analyze port conditions and classify them as either:
- Normal
- Congestion

It also provides an interactive Streamlit dashboard for data visualization and real-time user-input-based predictions.

**Note:** This project currently uses a synthetically generated dataset for development and demonstration.

## 2. Project Objectives

- Generate a dataset representing port operating conditions.
- Analyze factors associated with port congestion.
- Train a machine learning classification model.
- Predict congestion using user-provided inputs.
- Display results through an interactive dashboard.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Random Forest | Classification algorithm |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Joblib | Model serialization |
| Streamlit | Web dashboard |

## 4. Dataset Description

The project uses 500 synthetically generated records representing five ports:

- Chennai
- Mumbai
- Visakhapatnam
- Kolkata
- Cochin

### Dataset Features

| Feature | Description |
|---|---|
| port_name | Name of the port |
| vessel_arrivals | Number of vessel arrivals |
| port_capacity | Port handling capacity |
| queue_length | Number of vessels waiting |
| weather_severity | Weather severity score |
| historical_avg_delay | Historical average delay |
| hour_of_day | Hour of observation |
| month | Month of observation |
| delay_hours | Simulated delay in hours |
| congestion | Target classification |

The target variable is:
- 0 – No Congestion
- 1 – Congestion

## 5. Machine Learning Approach

The project uses a Random Forest Classifier.

### Data preprocessing
- One-hot encoding for the categorical port_name feature.
- Numerical features are passed through the preprocessing pipeline.
- Data is divided into training and testing sets.

### Training configuration
- Algorithm: Random Forest Classifier
- Number of estimators: 200
- Maximum depth: 10
- Training records: 400
- Testing records: 100
- Random state: 42

The target-derived delay_hours column is excluded from the model inputs to avoid target leakage.

## 6. Model Evaluation

The trained model achieved the following results on the held-out test dataset:

| Metric | Result |
|---|---:|
| Accuracy | 87.00% |
| Precision | 90.28% |
| Recall | 91.55% |
| F1-score | 90.91% |
| ROC-AUC | 92.67% |

### Confusion Matrix

```text
[[22  7]
 [ 6 65]]
```

Queue length and vessel arrivals were the most influential features according to the model's feature importance analysis.

These results describe performance on the synthetic test dataset and should not be interpreted as verified real-world port forecasting performance.

## 7. Project Structure

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
└── README.md
```

## 8. Installation and Setup

### Step 1: Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
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

Windows:

```bash
.venv\Scripts\activate
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Generate the dataset

```bash
python generate_data.py
```

### Step 7: Perform data analysis

```bash
python analyze_data.py
```

### Step 8: Train the model

```bash
python train_model.py
```

### Step 9: Launch the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

**Note:** The saved model and dataset must exist before launching the dashboard. The generation and training steps create these files.

## 9. Application Features

- Dataset overview and summary metrics.
- Congestion distribution visualization.
- Average delay comparison across ports.
- User-input-based congestion prediction.
- Congestion probability display.
- Dataset preview.

## 10. Limitations and Future Enhancements

### Current limitations
- The dataset is synthetically generated.
- The model has not been validated using real port operational data.
- Predictions depend on the quality and ranges of user-provided inputs.
- Model probabilities have not been independently calibrated.

### Future enhancements
- Integrate real-time port and vessel tracking data.
- Include live weather information.
- Improve model validation using real operational records.
- Explore additional machine learning algorithms.
- Add historical prediction tracking.
- Deploy the application to a cloud platform.

## 11. Conclusion

PortGuard AI demonstrates how machine learning can be used to classify port congestion conditions based on operational and environmental features.

The project combines synthetic data generation, exploratory data analysis, machine learning, model evaluation, and an interactive Streamlit dashboard into one end-to-end application.
