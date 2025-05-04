# Weather Prediction

This project uses two main models — LSTM and Transformer — to predict temperature and humidity. The application is built with Streamlit for easy deployment and interactive visualization.

## Table of Contents
- [Data Infomation](#data-information)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Result](#results)
## Data Information
### Daily data
- `temperature_2m_mean (°C)`: Mean daily temperature at 2 meters above ground.
- `temperature_2m_min (°C)`: Daily minimum temperature.
- `temperature_2m_max (°C)`: Daily maximum temperature.
- `relative_humidity_2m_mean (%)`: Mean daily relative humidity at 2 meters above ground.
### Hourly data
- `temperature_2m (°C)`: Hourly temperature at 2 meters above ground.
- `precipitation (mm)`: Hourly precipitation.
## Installation
1. Clone the repository:

```sh
git clone https://github.com/NamSeko/Weather_Prediction.git
cd Weather_Prediction
```

2. Install the required packages:

```sh
pip install -r requirements.txt
```
## Usage
To run the Streamlit application, use the following command:

```sh
streamlit run GUI/streamlit.py
```
## Model
- WeatherLSTM
- WeatherTransformer
## Results
### LSTM Model
|||`MAE`|`RMSE`|`R2`|
|-|-|-----|------|----|
|`Daily`|`Temperature`|2.09°C|2.56°C|0.89|
||`Relative Humidity`|6.70%|8.26%|0.55|
|`Hourly`|`Temperature`|2.05°C|2.41°C|0.91|
||`Relative Humidity`|4.55%|5.39%|0.89|
### Transformer Model
|||`MAE`|`RMSE`|`R2`|
|-|-|-----|------|----|
|`Daily`|`Temperature`|1.45°C|1.91°C|0.94|
||`Relative Humidity`|5.42%|6.90%|0.69|
|`Hourly`|`Temperature`|0.50°C|0.65°C|0.99|
||`Relative Humidity`|2.13%|2.99%|0.97|