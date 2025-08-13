
# AI-Driven Stock Market Prediction with GAN + LSTM

## Overview

This project explores deep learning for financial forecasting by implementing a **baseline LSTM model** and an **enhanced GAN + LSTM hybrid** for stock price prediction.

**LSTM (Long Short-Term Memory)**

LSTM networks are a type of recurrent neural network (RNN) designed to capture long-term dependencies in sequential data. They are particularly effective for time series forecasting because they can remember past information over many time steps, making them well-suited for predicting stock prices based on historical patterns.

**GAN (Generative Adversarial Network)**

GANs consist of two competing neural networks — a generator and a discriminator — that learn through adversarial training. The generator creates synthetic data, while the discriminator tries to distinguish between real and fake data. Over time, the generator produces increasingly realistic synthetic data.

**LSTM + GAN Hybrid**

In this project, the GAN is used to generate synthetic stock price sequences that mimic real market patterns. These synthetic sequences are combined with real historical data to train the LSTM model. This augmentation expands the dataset and introduces more variability, allowing the LSTM to generalize better and make more accurate predictions.

## Performance Improvement

**Baseline LSTM Model:** 
Achieved a Mean Absolute Error (MAE) of 5.98, RMSE of 7.32, and R² of 0.64.

**GAN + LSTM Model:** 
After enriching the training set with 100k+ synthetic sequences from the GAN, the LSTM achieved MAE = 0.4961, MSE = 0.5497, and R² = 0.9980 — a dramatic improvement in predictive accuracy.

By combining GANs and LSTMs, this approach benefits from LSTM’s ability to model sequential dependencies and GAN’s power to generate diverse, realistic training data, making it more robust for stock market prediction.

**Gen-AI Chatbot — Stock Buddy**

Stock Buddy is an AI-powered trading assistant that leverages a Retrieval-Augmented Generation (RAG) architecture to provide real-time, data-grounded market insights.

Built with LangChain for orchestration, it integrates Pinecone vector search to retrieve relevant market data, technical indicators, sentiment analysis, and GAN + LSTM model predictions. Retrieved context is passed to OpenAI GPT-4, which generates natural, explainable responses tailored to the user’s risk tolerance (conservative, moderate, aggressive).

**The chatbot can:**

Explain market conditions and technical signals.

Summarize model predictions with rationale.

Suggest buy/hold/sell actions.

Provide quick backtesting summaries.

This architecture ensures responses are both accurate and actionable, combining deep learning forecasts with conversational AI.

## Goals

**Accurate Stock Prediction** – Forecast short-term and medium-term stock price movements using advanced deep learning models.

**Data Augmentation with GANs** – Improve LSTM training by generating realistic synthetic stock sequences to enhance model generalization.

**Automated Technical Analysis** – Calculate and interpret indicators like RSI, MACD, and moving averages without manual intervention.

**Personalized Trading Strategies** – Adapt recommendations to user-defined risk profiles (conservative, moderate, aggressive).

**Conversational AI Insights** – Integrate predictions into a RAG-powered chatbot for real-time, explainable, and data-grounded investment advice.

---

## 🔍 Workflow

### 1. **Data Acquisition**

* Source: [Yahoo Finance](https://pypi.org/project/yfinance/) API (`yfinance` library)
* Historical data for **503+ tickers**
* Over 30 technical indicators collected

### 2. **Data Engineering & Storage**

* **Raw Data Storage**: AWS S3 (CSV & optimized Parquet)
* **Data Warehouse**: Snowflake (Processed & cleaned datasets)
* ETL pipeline built using Python, Pandas, and SQLAlchemy
* Ensured **data quality**: completeness, consistency, timeliness, uniqueness

### 3. **Exploratory Data Analysis (EDA)**

* Price action trends, support & resistance levels
* Volume analysis, moving averages, RSI, MACD, Bollinger Bands
* Visualizations built in Power BI for interactive analysis

### 4. **Feature Engineering**

* Computed **20+ technical indicators** (SMA, EMA, MACD, RSI, ATR, etc.)
* Normalized features using MinMaxScaler
* Created sequences for LSTM input (rolling window approach)

### 5. **Modeling**

#### **Base Model: LSTM**

* Sequence length: 60 days
* Train-test split: 80/20
* Metrics:

  * MAE: 5.98
  * RMSE: 7.32
  * R²: 0.64

#### **Fine-Tuned Model: GAN + LSTM**

* GAN generates **100k+ synthetic stock sequences**
* LSTM trained on enriched dataset (real + synthetic)
* Metrics:

  * MAE: 0.4961
  * MSE: 0.5497
  * R²: 0.9980

### 6. **Gen-AI Agent: Stock Buddy**

* Built with **LangChain** + **OpenAI GPT-4**
* Integrates with Pinecone for vector search
* Provides **real-time trading insights** based on technical indicators and user risk profile

---

## 🛠 Tools & Frameworks

* **Languages**: Python
* **ML/DL**: PyTorch, scikit-learn, LSTM, GAN
* **Data**: Pandas, NumPy, yfinance
* **Cloud**: AWS S3, AWS SageMaker, Snowflake
* **Gen-AI**: OpenAI GPT-4, LangChain, Pinecone
* **Visualization**: Power BI

---

## 📊 Results

| Model       | MAE   | MSE   | RMSE | R²     |
| ----------- | ----- | ----- | ---- | ------ |
| LSTM (Base) | 5.98  | 53.67 | 7.32 | 0.64   |
| GAN + LSTM  | 0.496 | 0.55  | 0.74 | 0.9980 |

---




