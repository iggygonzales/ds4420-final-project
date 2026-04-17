# DS4420 Final Project - Predicting National Renewable Energy Adoption

**DS 4420: Machine Learning and Data Mining 2 | Spring 2026**  
Ved Agrawal · Ben Pierce · Miguel Gonzales

[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-brightgreen)](https://ds4420-renewable-energy.streamlit.app/)

---

## Overview

This project investigates what economic and structural factors determine a country's share of renewable energy, and which countries are over- or under-performing relative to what their profile would predict.

We apply three complementary machine learning methods to the [Our World in Data World Energy Consumption dataset](https://ourworldindata.org/energy):

1. **Manual MLP** - a from-scratch neural network that predicts renewable share from structural features and identifies anomalous countries via residual analysis
2. **Bayesian Regression** - quantifies which features are reliable drivers of renewable adoption and with what degree of certainty
3. **ARIMA Time Series** - applied to a five-country set motivated by the largest MLP residuals, examining whether structural anomalies are growing or shrinking through 2030

---

## Data

**Source:** [Our World in Data - World Energy Consumption](https://ourworldindata.org/energy), also available via [Kaggle](https://www.kaggle.com/datasets/pralabhpoudel/world-energy-consumption).

The dataset consolidates country-level energy statistics from the BP Statistical Review of World Energy, the IEA, and Ember - covering 200+ countries and regions from 1900 to 2022 across 120+ variables.

**Final feature set (5 predictors):**

| Feature | Description |
|---|---|
| `gas_share_elec` | Share of electricity generated from natural gas |
| `fossil_elec_per_capita` | Per-capita fossil electricity generation |
| `coal_share_energy` | Coal share of total primary energy |
| `energy_per_gdp` | Energy consumption per unit of GDP |
| `energy_per_capita` | Total primary energy consumption per capita |

**Target:** `renewables_share_energy` - a country's share of total energy from renewables.

The 2018 cross-section was used for the MLP and Bayesian models (69 countries with complete data). ARIMA models use annual data from 1990–2022 for five selected countries.

---

## Models

### Model 1 - Manual MLP (Python / NumPy)

A two-hidden-layer neural network implemented from scratch, without any deep learning framework.

- **Architecture:** Input(5) → Dense(64, ReLU) → Dense(32, ReLU) → Output(1, linear)
- **Initialization:** He initialization
- **Optimizer:** SGD with momentum (`lr=0.005`, `momentum=0.9`)
- **Regularization:** L2 weight decay (`λ=0.01`) + early stopping (patience=300)
- **Evaluation:** Stratified 5-fold cross-validation (by renewable share quartile)
- **CV RMSE:** 9.03 ± 2.3 pp | **OOF R²:** 0.474

Out-of-fold residuals are used to rank countries as over- or under-performers. The largest outliers - Brazil (+23.9 pp), Norway (+14.7 pp), Morocco (−28.2 pp), and Poland (−25.1 pp) - motivate the ARIMA analysis.

### Model 2 - Bayesian Linear Regression (R / brms / Stan)

A Bayesian regression fitted on the same 2018 cross-section, using weakly informative priors and four MCMC chains via Stan.

- **Likelihood:** Gaussian
- **Priors:** `Normal(0, 5)` on coefficients; `StudentT(3, 0, 10)` on sigma and intercept
- **CV RMSE:** 9.24 ± 2.66 pp | **CV R²:** 0.402 ± 0.269

Key findings: `gas_share_elec`, `coal_share_energy`, and `fossil_elec_per_capita` are credibly negative predictors. `energy_per_capita` is credibly positive. `energy_per_gdp` remains uncertain. The Bayesian model provides posterior probabilities of over/under-performance for each country, complementing the MLP residual analysis.

### Model 3 - ARIMA Time Series (Python / pmdarima)

Country-level ARIMA models trained on annual renewable share from 1990–2022 for five countries selected from the MLP residual ranking.

**Countries:** Brazil, Norway, Morocco, Poland, United States

- **Order selection:** `auto_arima` over p, q ∈ {0,1,2,3}, d ∈ {0,1,2}, by AIC
- **Train:** 1990–2018 | **Test:** 2019–2022 | **Forecast:** through 2030
- **Test RMSEs:** 0.75 pp (US) to 3.41 pp (Brazil)

Selected orders: ARIMA(1,0,0) for Brazil, Norway, Morocco; ARIMA(0,2,1) for US; ARIMA(0,1,3) for Poland.

**2030 forecasts:** The US is the only country projected to grow meaningfully (+3.7 pp to 15.0%). Poland and Morocco remain on fossil-locked trajectories. Brazil and Norway show slight mean-reversion after recent peaks.

---

## Setup & Reproduction

### Python - MLP & ARIMA

Requires **Python 3.11**.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the MLP
python python/mlp_poc.py

# 4. Run ARIMA (Jupyter)
jupyter notebook Model3_ARIMA_Updated.ipynb
```

### R - Bayesian Regression

Requires **R** with the following packages: `brms`, `tidyverse`, `posterior`, `bayesplot`.

Stan is used as the backend sampler - see the [brms installation guide](https://paul-buerkner.github.io/brms/) for setup.

```r
# Open and knit the R Markdown file
rmarkdown::render("Model3_Bayes_Renewable_Energy1.Rmd")
```

A pre-rendered HTML report is available at `Model3_Bayes_Renewable_Energy1.html`.

---

## Results Summary

| Model | CV RMSE | CV R² |
|---|---|---|
| MLP (manual, NumPy) | 9.03 ± 2.3 pp | 0.474 |
| Bayesian Regression (brms) | 9.24 ± 2.66 pp | 0.402 |

The near-identical performance across both models suggests the five structural features capture most of the available signal in this 69-country sample, and that the remaining variance likely reflects missing variables (hydropower endowments, policy history, geographic resources) rather than modeling limitations.

---

## Live App

An interactive Streamlit app is available at:  
👉 **[ds4420-renewable-energy.streamlit.app](https://ds4420-renewable-energy.streamlit.app/)**

---

## References

1. H. Muhammad and M. T. Majeed, "The determinants of renewable energy production: A global study on panel data," *Pakistan Journal of Economic Studies*, vol. 7, no. 1, pp. 51–67, 2024.
2. D. S. Cristea et al., "Renewable energy strategy analysis in relation to environmental pollution for BRICS, G7, and EU countries," *Frontiers in Environmental Science*, vol. 10, 2022.
3. W. Ossai and T. M. Fagbola, "Machine learning-based predictive modelling of renewable energy adoption in developing countries," *Energy Reports*, vol. 14, pp. 66–84, 2025.
4. R. Basmadjian, A. Shaafieyoun, and S. Julka, "Day-ahead forecasting of the percentage of renewables based on time-series statistical methods," *Energies*, vol. 14, no. 21, 2021.
5. H. Ritchie, P. Rosado, and M. Roser, "Energy," Our World in Data, 2023. https://ourworldindata.org/energy
