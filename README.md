# DS4420 Final Project - Predicting National Renewable Energy Adoption

Predicting and analyzing renewable energy adoption across countries using three
independent models: a manual MLP neural network, Bayesian regression,
and ARIMA time series

---

## Setup

### Python (Model 1)

Requires Python 3.11.

**1. Create and activate a virtual environment**

```bash
# Mac/Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the model**

```bash
python python/mlp_poc.py
```

---


---

## Models

### Model 1 — MLP Neural Network 

Predicts each country's renewable electricity share from economic and
energy features. Implemented manually with NumPy 

- **Input:** 11 features (population, energy per capita, fossil/coal/gas share, etc.)
- **Architecture:** Input(11) > Dense(64, ReLU) > Dense(32, ReLU) > Output(1)
- **Training:** Mini-batch SGD with momentum, 2000 epochs
- **Output:** Per-country predictions and residuals, highlighting which countries
  over- or under-perform relative to their economic profile

### Model 2 — ARIMA Time Series 


### Model 3 — Bayesian Regression


---

## Data

**Source:** Our World in Data - [World Energy Consumption](https://www.kaggle.com/datasets/pralabhpoudel/world-energy-consumption?resource=download)

The dataset covers 200+ countries from 1900–2022 across 129 energy variables



