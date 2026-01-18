"""
Test functions to apply Surrogate Modeling enhancement preprocessing.
"""
#INITIALIZE THE LOCATIONS (HUC8) AND THE EVENT DETAILS
import fimserve

huc_id = '12030105'
event_date = '2016-01-01'
# boundary = '/Users/supath/Downloads/MSResearch/CNN/SurrogatModeling/SM_prediction/case_boundary/HUC12060102.shp'

# def test_prepare_forcing():
#     """
#     It now collects all the forcing from AWS S3 buckets for the given HUC and calculate the Low-Fidelity FIM based on the OWP HAND Model
#     based on the given event date using National Water Model (NWM) streamflow dataset.
#     """
#     fimserve.prepare_FORCINGs(
#         huc_id=huc_id, 
#         event_date=event_date, 
#         data='retrospective', 
#         # forecast_range=None, #For the forecasting Flood Inundation Mapping enhanceent [supports shortrange [18 lead hours], mediumrange [7 days], longrange [30 days]]
#         # forecast_date=None,  #If forecast_range is provided, forecast_date is required
#         # sort_by=None    #For Medium and Long range forecast, to get the streamflow data, sort_by can be 'maximum', 'minimum', 'median']
#         )
#     print("FORCING data preparation completed.")
    
# def test_applySM():
#     #Enhance the LOW-Fidelity FIM
#     fimserve.enhanceFIM(huc_id)
#     print("Surrogate Model based FIM enhancement completed.")
    
def test_get_exposure():
    fimserve.getpopulation_exposure(huc_id)
    print("Population exposure estimation completed.")
    
    fimserve.getbuilding_exposure(huc_id, geeprojectID='supathdh')
    print("Building exposure estimation completed.")
    