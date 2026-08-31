import fimserve as fm

huc = "03020202"


# Run the test function
def test_runfim():

    # Download the data
    fm.DownloadHUC8(huc, version="4.8")

    # # #Downloading the raster without headwaters
    # # fm.DownloadHUC8(huc, stream_order=[5, 6, 7, 8, 9, 10])

    # #Hindcast data
    # Get the NWM data
    start_date = "2016-10-01"
    end_date = "2016-10-10"

    # #For 12060202
    feature_id = ["5513784", "5513550", "5512092", "5512484"]
    usgs_sites = ["08096500", "08096580", "08092000", "08091000"]

    # # #Similarly, for 12060102
    # # feature_id= ['5489963', '5488917', '5489939']
    # # usgs_sites = ['08084200', '08083100', '08083240']

    # # #For 03020202
    # # feature_id = ['11239079', '11239241', '11239465', '8791643']
    # # usgs_sites = ['0209205053', '02091814', '02089500', '02089000']

    # for fixed date or day data
    value_times = ["2024-10-05"]
    # fm.getNWMretrospectivedata(huc, value_times)
    fm.getNWMretrospectivedata(huc, start_date, end_date, discharge_sortby="maximum")

    # # #Get USGS data
    # fm.getUSGSsitedata(huc, start_date, end_date)

    # fm.plotNWMStreamflow(huc, start_date, end_date, feature_id)
    # # #Get the forecast data
    # # #Short range forecast
    # fm.getNWMForecasteddata(
    #     huc, forecast_range="shortrange", forecast_date="2024-11-14"
    # )

    # # #Long range forecast
    # # fm.getNWMForecasteddata(
    # #     huc, forecast_range="longrange", forecast_date="2024-11-14", hour=6
    # # )

    # # #Medium range forecast
    # fm.getNWMForecasteddata(
    #     huc, forecast_range="mediumrange", forecast_date="2024-11-14", hour=6
    # )

    # # #Analysis and Assimilation (AnA) data, indexed by valid time instead of a forecast cycle. Available from 2018-09-17, so the 2016 range above is too early
    # # #One aggregated discharge file over the range
    # fm.getNWManalysisAssim(huc, start_date, end_date)

    # # #Continuous hourly discharge, one CSV per timestep
    # fm.getNWManalysisAssim(huc, start_date, end_date, continuous_discharge=True)

    # #Only the event day, or only a single timestep, within the range
    fm.getNWManalysisAssim(huc, start_date, end_date, value_times)
    # fm.getNWManalysisAssim(
    #     huc, start_date, end_date, value_times="2024-09-27 12:00:00"
    # )

    # Run the FIM model
    fm.runOWPHANDFIM(huc, depth=True)
