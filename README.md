# Portfolio: Computational Economics & Data Science
**Candidate:** Lok In Ho


## Overview
This repository contains my core computational work samples, demonstrating proficiency in **Big Data Construction**, **Machine Learning**, **Econometric Simulation**, and **Agent-Based Modelling**.

These files were selected to demonstrate the technical skills required for the Nuffield-funded "Towards housing affordability" project.

---

## 📂 File 1: Data Scaping, Simulation & NLP
**Filename:** `Data Scaping, 3D-model - Differences of predicting power between underfit, good-fit and over-fit models, NLP and Classification.qmd`

This document contains three distinct computational projects:

### 1. Automated Data Scraping (Database Construction)
* **Objective:** Construct a structured database of prices and ratings from unstructured web sources.
* **Tech Stack:** R (`rvest`, `httr`).
* **Relevance:**
    * Implemented **ethical scraping protocols** (checking `robots.txt`) and automated error handling (retries/delays).
    * Demonstrates the ability to "construct databases" and clean Big Data required for planning documents.

### 2. NLP & Classification
* **Objective:** Classify stance in climate change discourse ("Believer" vs "Sceptic").
* **Tech Stack:** R (`quanteda`, `glmnet`).
* **Relevance:**
    * Compared Top-Down (Dictionary) vs. Bottom-Up (TF-IDF) approaches.
    * Trained a **Lasso Logistic Regression** model with 5-fold Cross-Validation (AUC = 0.73).
    * Directly addresses the desirable skill for **"classification and text analysis using machine learning"**.

### 3. Econometric Modelling (Bias-Variance Trade-off)
* **Objective:** Analyse the risks of overfitting vs. underfitting using Monte Carlo simulation.
* **Tech Stack:** R (Base plotting, Simulation).
* **Relevance:**
    * Implemented **k-fold Cross-Validation** to minimize Mean Squared Error (MSE).
    * Visualized 3D prediction surfaces using `persp` plots to assess model generalizability.

---

## 📂 File 2: Experimental Design & Analysis
**Filename:** `Quantitative measure(ANOVA) .Rmd`

* **Objective:** Rigorous statistical hypothesis testing on behavioural data.
* **Tech Stack:** R (`afex`, `emmeans`, `ggplot2`).
* **Key Details:**
    * Conducted **3-Way Mixed-Design ANOVAs** to analyse interactions between bias, reference style, and role type.
    * Visualized complex three-way interactions using error-bar plots for clear interpretation.

---

## 📂 File 3: Complex Systems Modelling
**Filename:** `Networking and Agent based Model.qmd`

* **Objective:** Simulation of complex adaptive systems and agent interactions.
* **Tech Stack:** R / Agent-Based Frameworks.
* **Relevance:** Demonstrates advanced computational modelling capabilities applicable to housing market dynamics and supply-chain network effects.
  
## 📂 File 4: Automated Data Analysis Pipeline & API Integration (Python)

**Filename:** `behavioural_financial_analysis.py`

This project demonstrates a full data engineering and analysis pipeline, automating the retrieval and statistical validation of behavioural datasets.

* **Objective:** Construct a unified dataset from remote servers and local files for behavioural financial analysis.
* **Tech Stack:** Python (`pandas`, `scipy`, `requests`, `matplotlib`).
* **Relevance:**
    * **Automated Data Retrieval:** Connects to **Supabase REST API** to fetch live asset data using secure header authentication.
    * **Data Cleaning:** Automated the merging of personality CSVs with JSON asset data, handling missing values and filtering for specific currencies (GBP).
    * **Statistical Analysis:** Conducted **Pearson Correlation** and **Linear Regression** analyses to test hypotheses linking personality traits (e.g., Risk Tolerance) to financial assets.

**2. Data Cleaning & Pre-processing**
* **Objective:** Prepare heterogeneous data for rigorous statistical testing.
* **Tech Stack:** Python (`pandas`).
* **Relevance:**
    * Automated the merging of personality CSVs with JSON asset data on unique IDs.
    * Implemented filtering logic to isolate specific currency holdings (GBP) and handle missing values.
    * Directly addresses the "Data transcription, clearing, and pre-processing" criterion.

**3. Statistical Analysis & Visualisation**
* **Objective:** Validate hypotheses connecting personality traits to financial risk.
* **Tech Stack:** Python (`scipy.stats`, `matplotlib`).
* **Relevance:**
    * Conducted **Pearson Correlation** and **Linear Regression** analyses to test behavioural hypotheses.
    * Generated automated scatter plots to visualize the relationship between Risk Tolerance and Asset Value.
---
*For any questions regarding the code or methodology, please refer to my CV.*
