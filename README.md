# Solar PV Power Generation Forecast

This project analyzes solar plant operational and weather data to predict AC power output using machine learning models. The workflow merges generation and sensor datasets, cleans the data, engineers time-based features, trains several regression models, and compares their performance.

## Project goals

- Predict AC power generation from environmental and time features
- Compare model accuracy across multiple regression approaches
- Produce a summary CSV with model performance metrics

## Data

The project uses these files:

- `Plant_2_Generation_Data.csv`
- `Plant_2_Weather_Sensor_Data.csv`

These files contain timestamped generation and weather readings for the solar plant. The script merges them on `DATE_TIME` and filters invalid or incomplete records before training.

## Models evaluated

The pipeline compares the following models:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Baseline Mean Predictor

Performance is evaluated using:

- R² Score
- MAE
- RMSE

## Output

The project saves a comparison summary to:

- `model_comparison.csv`

This file contains the model metrics in ranked order.

## Requirements

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn
```

## Usage

Run the main pipeline from the project root:

```bash
python run_ac_power_pipeline.py
```

This script:

1. Loads the generation and weather data
2. Cleans invalid timestamps and missing values
3. Merges the datasets by datetime
4. Builds time-based features
5. Splits data chronologically into train/test sets
6. Trains and evaluates the regression models
7. Saves the performance comparison to `model_comparison.csv`

## Repository structure

```text
.
├── README.md
├── .gitignore
├── run_ac_power_pipeline.py
├── model_comparison.csv
├── Plant_2_Generation_Data.csv
├── Plant_2_Weather_Sensor_Data.csv
├── solar-power-generation-forecast_executed.ipynb
├── solar-power-generation-forecast[1](1).ipynb
└── .ipynb_checkpoints/
```

## Notes

- The notebooks provide exploratory analysis and visualizations related to the same forecasting workflow.
- The model training script is the main reproducible pipeline for generating the final comparison metrics.
