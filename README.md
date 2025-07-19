# 🌍 **Satellite-Derived Species Distribution Model (SatSDM)**

> ⚠️ **Important Installation Requirement**:
> Before using this QGIS plugin, you must install the following Python packages using **OSGeo4W Shell**:
> 
> 1. **Open OSGeo4W Shell as Administrator** (right-click and select "Run as administrator").
> 2. Run the following command to install the required packages:
> 
> ```bash
> pip install scikit-learn rasterio
> ```
> These dependencies are essential for the plugin to function properly. Make sure to run the `OSGeo4W Shell` with administrator privileges to install these packages.

## 🚀 **v1.0.0 - Initial Release**

This is the first official release of the **SatSDM** plugin, designed to facilitate species distribution modeling using satellite data. It provides a streamlined workflow for fetching species presence data, processing environmental layers, and running Maxent models for species habitat prediction.

---

## ✨ **Features**:

### 1. **📊 Model Input Tab**:
- **Presence Data**:
  - Fetch presence data from **GBIF** 🌐 or upload data from a **local file** 📂.
  - Enter **species name** 🐾 and **country code** 🌍 to retrieve data. 
    - Example: For **Korea**, use the country code **'KR'** 🇰🇷.
    - Users can also obtain data from specific **Korea National Park Service** 🏞️ datasets (Korea, Republic of).
- **Environmental Layers**:
  - Add and remove environmental layers 🌿 to improve model accuracy.

### 2. **🌌 SD Layers Tab**:
- **Spectral Layer from Satellite Images**:
  - Select from various **spectral indices** (NDVI, NDWI, SAVI, etc.) derived from satellite images 🌍.
  - Choose satellite source (e.g., **KOMPSAT-3**) 📸.
- **Terrain Layer from DEM**:
  - Automatically calculate terrain indices like **Slope**, **Aspect**, and **Topographic Position Index (TPI)** 🏞️ from DEM files.
- **Proximity Layer from Land Cover**:
  - Create proximity layers based on **land cover data** 🏙️.

### 3. **⚙️ Model Manager Tab**:
- **Feature Types**:
  - Select feature types like **Linear**, **Hinge**, **Product**, and **Threshold** for the model 🔧.
- **Regularization & Advanced Options**:
  - Set **Beta Multiplier**, **Prevalence**, **Convergence Tolerance**, and more to refine your model 🔍.

### 4. **📊 Model Report Tab**:
- **Model Evaluation**:
  - View model performance metrics such as **AUC**, **ROC**, **Precision**, **Recall**, and **F1-score** 📈.
- **ROC Curve**:
  - Visualize the **ROC curve** 📉 to evaluate model accuracy.

---
