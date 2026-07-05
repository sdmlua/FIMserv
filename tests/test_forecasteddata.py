import fimserve as fm

huc = "03020202"

# mention the forecast range as shortrange or mediumrange or longrange
# Download the data if user wants to download the data for fixed date and hour otherwise it will download the data by taking current date and time fore forecasted data
# fm.getNWMForecasteddata(
#     huc, forecast_range="shortrange", forecast_date="2021-09-01", hour=6
# )

# For current date and time forecasted data
fm.getNWMForecasteddata(huc, forecast_range="shortrange")


# By default the data will be filtered based on maximum discharge value
# Otherwise user need to mention

# fm.getNWMForecasteddata(huc, forecast_range="longrange", sort_by="minimum")
