# 🏠 India Property Investment Advisor

**ML-powered Real-Estate Valuation & Investment Decision App for Indian Markets**

[🔗 Live App](https://india-property-investment-advisor.streamlit.app/)

---

## 📌 Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Data Overview](#data-overview)
- [Modeling](#modeling)
- [App Features](#app-features)
- [How to Use (Non-Technical Users)](#how-to-use-non-technical-users)
- [Project Structure](#project-structure)
- [Installation & Running Locally](#installation--running-locally)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## 🧠 Introduction

This project enables **homebuyers and investors in India** to make **data-driven property investment decisions**. With machine learning models under the hood and a sleek **Streamlit** UI, the app instantly tells users whether a listing is a **GOOD** investment or **RISKY**, and estimates a **fair market price**.

It brings transparency, valuation intelligence, and investment foresight to the chaotic Indian real estate market.

---

## 🚨 Problem Statement

Property buyers in India often face:

- ❌ Overpriced listings
- ❌ Poor understanding of locality quality
- ❌ No objective way to assess deal quality
- ❌ Uncertainty in future appreciation

**This app solves that by offering:**

✔ Fair price prediction  
✔ Investment quality classification  
✔ Growth-based projection  
✔ Clear verdict: **GOOD** or **RISKY**

---

## 📊 Data Overview

The app uses a **synthetic real-estate dataset** with **250,000 records**, including:

| Feature                  | Description                                |
|--------------------------|--------------------------------------------|
| City, Locality           | Location indicators                        |
| Property Type            | Apartment / House / Villa                  |
| BHK, Size (SqFt)         | Property configuration                     |
| Age of Property          | In years                                   |
| Nearby Schools/Hospitals | Infra & accessibility metrics              |
| Asking Price (Lakhs)     | Seller's price                             |
| Score (0–7)              | Derived investment quality score           |
| Good_Investment          | Target label: 1 = Good, 0 = Risky          |

**Label Distribution**  
- ✅ Good Investment: **27%**  
- ⚠️ Risky Investment: **73%**  
(Realistic—most Indian listings are overpriced)

---

## 🤖 Modeling

### 1️⃣ Classification Model

- **Goal:** Predict if the property is a good investment.
- **Model:** Logistic Regression  
- **Pipeline Includes:**
  - One-Hot Encoding
  - Standard Scaling
  - Class balancing
- **Performance:**
  - Accuracy: ~90%
  - ROC-AUC: ~0.93
  - F1-Score: ~0.84

### 2️⃣ Regression Model

- **Goal:** Predict the fair market price (Lakhs).
- **Model:** Random Forest Regressor
- **Performance:**
  - MSE: ~1.27
  - RMSE: ~1.12
  - MAE: ~0.80 Lakhs (~₹80,000 error)
  - R²: 0.9999

---

## 🖥️ App Features

Users input:

- Location, BHK, Size, Age
- Property Type
- Nearby Schools & Hospitals
- Asking Price
- Growth Rate & Horizon

The app outputs:

- **🏷 Investment Verdict**: GOOD or RISKY
- **📊 Probability** of being a good investment
- **💰 Fair Market Price**
- **📉 Over/Under Value Explanation**
- **📈 5-Year Value Projection**
- **🔍 Debug View** of input processing (optional)

---

## 🕹️ How to Use (Non-Technical Users)

1. **Enter city and locality**  
   Example: `"Hyderabad – Madhapur"`

2. **Fill in property details**  
   → BHK, SqFt, Age, Property Type

3. **Enter nearby infra details**  
   → Schools, hospitals (higher = better)

4. **Set asking price**  
   → App compares it to fair value

5. **Adjust growth assumptions**  
   → For long-term projections

6. **Click “Evaluate Investment 🚀”**  
   → Get instant verdict, value insights & growth forecast

---

## 📁 Project Structure

```bash
India_Property_Investment_Advisor/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   │   ├── train_classification.py
│   │   ├── train_regression.py
│   │   ├── predict.py
│   │   └── saved pipelines (.pkl)
│   └── app/
│       └── streamlit_app.py
│
├── models/
├── requirements.txt
├── streamlit_app.py
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
streamlit run streamlit_app.py
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

