

# 🏠 India Property Investment Advisor

**ML-powered Real Estate Valuation & Investment Decision Platform for Indian Markets**

[🔗 Live App](https://india-property-investment-advisor.streamlit.app/)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Data Overview](#data-overview)
- [Modeling Approach](#modeling-approach)
- [Application Features](#application-features)
- [Insights Dashboard](#insights-dashboard)
- [How to Use (Non-Technical Users)](#how-to-use-non-technical-users)
- [Project Structure](#project-structure)
- [Installation & Running Locally](#installation--running-locally)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## 🧠 Overview

**India Property Investment Advisor** is an end-to-end **machine learning–driven decision support system** that helps homebuyers and real estate investors evaluate whether a property listing is a **GOOD investment** or **RISKY**, and estimates its **fair market value**.

The project combines:
- Predictive modeling
- Business-driven feature engineering
- An interactive **Streamlit application**
- A full **market insights dashboard**

to bring transparency and data-backed intelligence to Indian real estate decisions.

---

## 🚨 Problem Statement

Real estate decisions in India are often driven by:
- ❌ Emotional judgment
- ❌ Overpriced listings
- ❌ Lack of locality-level insights
- ❌ No objective way to assess investment quality

### This project addresses these gaps by providing:
- ✔ Fair price estimation
- ✔ Investment quality classification
- ✔ Over / under-valuation analysis
- ✔ Data-backed market insights
- ✔ Clear, explainable outputs for non-technical users

---

## 📊 Data Overview

The application is built on a **synthetic real estate dataset with 250,000 property records**, designed to reflect realistic Indian market behavior.

### Key Features

| Feature                  | Description |
|--------------------------|-------------|
| City, Locality           | Location indicators |
| Property_Type            | Apartment / Independent House / Villa |
| BHK, Size_in_SqFt        | Property configuration |
| Age_of_Property          | Property age (years) |
| Nearby_Schools           | Schools within 5 km |
| Nearby_Hospitals         | Hospitals within 5 km |
| Price_in_Lakhs           | Asking price |
| Score (0–7)              | Derived investment quality score |
| Good_Investment          | Target label (1 = Good, 0 = Risky) |

### Label Distribution

- ✅ **Good Investment:** ~27%
- ⚠️ **Risky Investment:** ~73%

> This skew reflects real-world markets where most listings are not optimal investment opportunities.

---

## 🤖 Modeling Approach

### 1️⃣ Investment Classification Model

- **Objective:** Predict whether a property is a *good investment*
- **Model:** Logistic Regression
- **Pipeline Includes:**
  - One-hot encoding for categorical features
  - Feature scaling
  - Class imbalance handling
- **Performance Metrics:**
  - Accuracy: ~90%
  - ROC-AUC: ~0.93
  - F1-Score: ~0.84

---

### 2️⃣ Price Valuation Model

- **Objective:** Estimate fair market price (₹ Lakhs)
- **Model:** Random Forest Regressor
- **Performance Metrics:**
  - RMSE: ~1.12 Lakhs
  - MAE: ~0.80 Lakhs
  - R² Score: ~0.999

> Both models are saved as reusable pipelines and loaded directly into the Streamlit app.

---

## 🖥️ Application Features

### 🔹 Property Evaluation (Prediction App)

Users can input:
- City & Locality
- Property Type, BHK, Size
- Age of Property
- Nearby Infrastructure
- Asking Price
- Growth Rate & Investment Horizon

**Outputs:**
- ✅ Investment Verdict (GOOD / RISKY)
- 📊 Probability of being a good investment
- 💰 Model-estimated fair price
- 📉 Over / Under-valuation explanation
- 📈 Growth-based future price projection
- 🔍 Optional debug view of model inputs

---

## 📊 Insights Dashboard

A dedicated **Insights Dashboard** built using the **full dataset (250k rows)** provides market-level intelligence:

### Dashboard Highlights
- Average property prices (₹ Lakhs)
- Average price per SqFt
- Good investment rate
- Listings count
- Price distribution
- City-wise price analysis
- Property type distribution
- Investment quality by city & property type

### Interactive Filters
- City (All / specific)
- Property Type (All / selected)
- BHK
- Price range
- Size range
- Age of property
- Toggle: *Show only good investments*

This dashboard helps users and stakeholders **understand patterns**, not just individual predictions.

---

## 🕹️ How to Use (Non-Technical Users)

1. Open the **Live App**
2. Enter property details:
   - City & locality
   - BHK, size, age
3. Add nearby infrastructure details
4. Enter asking price
5. Adjust growth assumptions if needed
6. Click **“Evaluate Investment 🚀”**
7. Review:
   - Investment verdict
   - Fair price
   - Over / under-valuation insight
8. Switch to **Insights Dashboard** for market trends

---

## 📁 Project Structure

```bash
India_Property_Investment_Advisor/
│
├── data/
│   ├── raw/
│   └── processed/
│       └── india_housing_with_targets.csv
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   │   ├── train_classification.py
│   │   ├── train_regression.py
│   │   ├── predict.py
│
├── models/
│   ├── classifier_pipeline.pkl
│   └── regression_pipeline.pkl
│
├── pages/
│   └── 01_Property_Market_Insights.py
│
├── Property_Investment_Advisor.py
├── requirements.txt
└── README.md
```
---

## 🚀 Installation & Running Locally

### 1. Clone the Repo

```bash
git clone https://github.com/mani9kanta3/India_Property_Investment_Advisor.git
cd India_Property_Investment_Advisor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit App

```bash
streamlit run Property_Investment_Advisor.py
```

## 🌟 Future Improvements

- 📌 **Integrate real price data** from Delhi / Mumbai / Bangalore  
- 🌲 **Use XGBoost** for stronger classification performance  
- 📈 **Add time-series forecasting** for price appreciation trends  
- 🗺️ **Integrate maps & heatmaps** for visual property insights  
- 📱 **Build APIs** for mobile and web app integration  

## 👤 Author

**Manikanta Pudi**  
_Data Analyst_  
🔗 GitHub: [mani9kanta3](https://github.com/mani9kanta3)


