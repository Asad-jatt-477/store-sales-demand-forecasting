<h1 align="center">Retail Demand Forecasting System</h1>

<p align="center">
  Daily demand forecasting for <b>1,782 store × product-family series</b> of Corporación Favorita, a grocery chain in Ecuador,<br>
  built as a leakage-safe time-series pipeline and deployed as a live what-if forecasting dashboard.
</p>

<p align="center">
  <a href="https://favorita-demand-forecast.vercel.app" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%9A%80%20LIVE%20DEMO-Open%20the%20Dashboard-0E5A3E?style=for-the-badge&labelColor=0A3F2C" alt="Live Demo" height="45">
  </a>
</p>

---

## Overview

The **Retail Demand Forecasting System** predicts how many units of each product family will sell in each of 54 stores on every one of the next 16 days. It is built on 4.5 years of daily sales (January 2013 to August 2017) from the Kaggle *Store Sales – Time Series Forecasting* dataset.

The project covers the complete journey from raw data to a working product: data preparation, exploratory analysis, outlier review, feature engineering, model comparison, hyperparameter tuning, a final forecast and a business evaluation. The trained model is served through a live dashboard, where a planner can change the promotion plan or mark a holiday and see the forecast recalculated instantly.

| Result | Model | Seasonal baseline |
|---|:---:|:---:|
| Forecast error on an untouched holdout window (WAPE) | **11.73%** | 13.73% |
| Forecast error on the validation window (WAPE) | **11.45%** | 18.31% |
| Over-ordered units (6 backtest windows) | **4.96M** (23% fewer) | 6.44M |
| Units missing from the shelf (6 backtest windows) | **5.12M** (24% fewer) | 6.73M |

The model is more accurate than the seasonal baseline in **all 33 product families** and **all 54 stores**.

## Aim

To build a **reliable 16-day demand forecast that supports day-to-day ordering decisions**, one that:

- uses only information that is genuinely available on the day the forecast is made;
- is tested honestly against a strong seasonal baseline on past periods the model never saw;
- expresses its accuracy in business terms: excess stock, missed demand and the safety stock each product family needs;
- can be used by non-technical planners through a simple, interactive web application.

## Problem

A grocery chain has to decide every day how much of each product to order for each store. Ordering too much ties up money and, for fresh food such as produce, dairy, meat and bakery, turns into waste. Ordering too little leaves shelves empty and loses sales.

Getting this right is difficult because demand is shaped by many forces at once:

- **Scale:** 54 stores × 33 product families means 1,782 separate daily sales series to forecast.
- **Rhythms:** sales rise strongly at weekends, around the start of the month, and in December.
- **Promotions and holidays:** both shift demand, but differently for each product family and store.
- **Irregular series:** some product families sell nothing on most days, while others sell thousands of units daily.
- **Unusual events:** the April 2016 earthquake, store closures and product families that were not recorded in the early years distort the history.

Simple rules of thumb, such as repeating the average of the same weekday in recent weeks, cannot capture all of this at once. The goal is a forecast that does, and that can be trusted because it has been tested the way it will be used: predicting days it has never seen.

## Features

- **Clean daily foundation:** a complete calendar for every store and product family, with store closures, store openings and holidays joined in, and training restricted to the period from June 2015, when every product family was being recorded.
- **Evidence-based outlier review:** 8,267 extreme sales days were traced to real causes such as company-wide special days, promotion surges and stock-outs. Capping and replacing them was tested and did not reduce error, so the real history was kept.
- **Leakage-safe features:** 15 inputs covering store, product family, calendar, national holidays and events, items on promotion, and sales history from 16 to 364 days back. Every sales input is at least 16 days old, so the forecast never relies on information it could not have had.
- **Strict validation:** time-based backtests on six past 16-day windows, with one holdout window kept completely untouched until the final evaluation.
- **Model selection and tuning:** LightGBM, XGBoost, HistGradientBoosting and a linear model were compared against two baselines, then tuned with Optuna. A tuned HistGradientBoosting model was selected.
- **Business evaluation:** errors broken down by product family, store, store type, forecast day, holidays and promotions, plus excess-stock and missed-demand totals and per-family safety buffers for 80% and 90% coverage.
- **Live forecast service:** a Python API that loads the trained model and returns the forecast, the what-if scenario and recommended order quantities.
- **Interactive dashboard:** seven pages covering the forecast overview, a what-if forecast studio, backtests, data insights, how the model works, business impact, and method and limitations.

## Architecture Diagram

<p align="center">
  <img src="docs/architecture.png" alt="System Architecture — Retail Demand Forecasting System" width="100%">
</p>

<p align="center">
  <img src="docs/screenshots/overview.png" alt="Dashboard overview" width="49%">
  &nbsp;
  <img src="docs/screenshots/forecast-studio.png" alt="Forecast studio: live what-if forecast" width="49%">
</p>
<p align="center"><sub>Left: dashboard overview. Right: forecast studio with a live what-if scenario.</sub></p>

## Benefits

- **Less waste and fewer empty shelves:** compared with the seasonal rule of thumb, over-ordered units fell by 23% and missing units by 24% across six past periods.
- **Ready-to-use order quantities:** per-family safety buffers turn each forecast into an order that covers demand on 80% or 90% of days.
- **Results that can be trusted:** every reported figure comes from periods the model never saw while it was being built.
- **Scenario planning:** planners can test a promotion or a holiday for any store and product family before committing to an order.
- **Transparent limitations:** the dashboard shows where the forecast is weaker, such as local holidays, very small series and periods of fast growth.
- **Instant access:** anyone can explore the forecast in the browser, with no installation required.

## Conclusion

The Retail Demand Forecasting System shows a complete, evidence-driven approach to demand forecasting in retail. Every decision, from how unusual days were handled to which features and model settings were kept, was tested on the data before it was accepted, and the final result was measured on a period that played no part in building the model. With **11.73% error on the untouched holdout against 13.73% for the seasonal baseline**, higher accuracy in every product family and every store, and a live dashboard that turns forecasts into order quantities, it is a practical, deployment-ready foundation for better ordering decisions.

<p align="center">
  <a href="https://favorita-demand-forecast.vercel.app" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%9A%80%20TRY%20THE%20LIVE%20DEMO-favorita--demand--forecast.vercel.app-0E5A3E?style=for-the-badge&labelColor=0A3F2C" alt="Try the Live Demo" height="40">
  </a>
</p>
