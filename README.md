# Biodiversity-Challenge
The objective of this data challenge is to predict the presence, or non-presence, of frog species at given locations in southeastern Australia.

---

## Table of Contents

- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Installation & Dependencies](#installation--dependencies)
- [Data Preprocessing](#data-preprocessing)
  - [Loading & Exploring Data](#loading--exploring-data)
  - [Extracting Environmental Variables](#extracting-environmental-variables)
  - [Data Cleaning](#data-cleaning)
- [Modeling](#modeling)
  - [Model Comparison](#model-comparison)
  - [Best Model Selection](#best-model-selection)
- [Generating Submission](#generating-submission)
- [Model Performance](#model-performance)
- [Acknowledgments](#acknowledgments)


---

## Overview

The project aims to predict frog species presence using environmental data from southeastern Australia. The workflow includes:

1. **Data Loading:**  
   Loading training data containing geographic coordinates and occurrence status.
2. **Environmental Variable Extraction:**  
   Extracting 14 environmental variables from a GeoTIFF file.
3. **Data Cleaning:**  
   Removing duplicate entries and preparing the dataset for modeling.
4. **Model Training:**  
   Comparing five different classification models (Logistic Regression, SVM, Decision Tree, Random Forest, and Naive Bayes).
5. **Prediction Generation:**  
   Applying the best-performing model to new data and generating predictions.

---

## Dataset Description

The dataset includes:
- **Train Dataset (`Training_Data.csv`):** Contains geographic coordinates (`Latitude`, `Longitude`) and `Occurrence Status` (0 for absence, 1 for presence) for 6312 locations.
- **Environmental Data (`TerraClimate_output.tiff`):** A GeoTIFF file containing 14 environmental variables across the study area.
- **Validation Data (`Validation_Template.csv`):** Contains geographic coordinates for 2000 locations where predictions need to be made.

---

## Installation & Dependencies

Ensure you have Python 3 and the following libraries installed:

-   **numpy**
-   **pandas**
-   **xarray**
-   **matplotlib**
-   **seaborn**
-   **geopandas**
-  **rasterio**
-   **rioxarray**
-   **scikit-learn**
-   **tqdm**

Install them via pip:

```bash
pip install numpy pandas xarray matplotlib seaborn geopandas rasterio rioxarray scikit-learn tqdm
```

---

## Data Preprocessing

### Loading & Exploring Data

The training data is loaded and inspected:

```
ground_df = pd.read_csv("Training_Data.csv")
ground_df.shape  # (6312, 3)
```

### Extracting Environmental Variables

Environmental variables are extracted from the GeoTIFF file using rasterio:

```
with rasterio.open(tiff_path) as src1:
    variable1 = src1.read(1)  # variable [aet]
    variable2 = src1.read(2)  # variable [def]
    # ... (other variables)
    variable14 = src1.read(14)  # variable [ws]
```

These variables include:

-   Actual evapotranspiration (aet)

-   Deficit (def)

-   Palmer Drought Severity Index (pdsi)

-   Potential evapotranspiration (pet)

-   Precipitation (ppt)

-   Discharge (q)

-   Soil moisture (soil)

-   Solar radiation (srad)

-   Snow water equivalent (swe)

-   Minimum temperature (tmin)

-   Maximum temperature (tmax)

-   Vapor pressure (vap)

-   Vapor pressure deficit (vpd)

-   Wind speed (ws)

### Data Cleaning

Duplicate entries are removed based on the environmental variables and occurrence status:

```
columns_to_check = ['aet', 'def', 'pdsi', 'pet', 'ppt', 'q', 'soil', 'srad', 'swe', 'tmin', 'tmax', 'vap', 'vpd', 'ws', 'Occurrence Status']
final_data = final_data.drop_duplicates(subset=columns_to_check, keep='first')
```

---

## Modeling

### Model Comparison

Five classification models are compared:

1.  Logistic Regression

2.  Support Vector Machine (SVM) with RBF kernel

3.  Decision Tree

4.  Random Forest

5.  Naive Bayes

The models are trained and evaluated using an 80-20 train-test split with stratification.

### Best Model Selection

The model with the highest accuracy on the test set is selected. In this case, the SVM model achieved the highest accuracy of 72.69%.

---

## Generating Predictions
The best-performing model (Logistic Regression) is applied to the validation data to generate predictions:

```
final_prediction_series = apply_best_model(bool_lr, bool_svm, bool_dt, bool_rf, bool_nb, model_data, final_val_data)
submission_df = pd.DataFrame({'Latitude':test_file['Latitude'].values, 'Longitude':test_file['Longitude'].values,  'Occurrence Status':final_prediction_series.values})
submission_df.to_csv("Predicted_Data.csv",index = False)
```

---

## Model Performance
The best model (SVM) achieved an accuracy of 72.69% on the test set with the following classification report:

```
              precision    recall  f1-score   support

           0       0.74      0.64      0.69       741
           1       0.72      0.80      0.76       845

    accuracy                           0.73      1586
   macro avg       0.73      0.72      0.72      1586
weighted avg       0.73      0.73      0.72      1586
```

---

## Acknowledgments

This project uses environmental data from the TerraClimate dataset. The machine learning implementation is powered by open-source libraries including Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn.
