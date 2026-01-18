import warnings

from fimserve.fimevaluation.fims_setup import fim_lookup

warnings.simplefilter("ignore")

from .datadownload import DownloadHUC8
from .streamflowdata.nwmretrospective import getNWMretrospectivedata
from .runFIM import runOWPHANDFIM

from .streamflowdata.forecasteddata import getNWMForecasteddata
from .streamflowdata.geoglows import getGEOGLOWSstreamflow

# plots
from .plot.nwmfid import plotNWMStreamflow
from .streamflowdata.usgsdata import getUSGSsitedata
from .plot.comparestreamflow import CompareNWMnUSGSStreamflow
from .plot.usgs import plotUSGSStreamflow
from .plot.src import plotSRC

# For table
from .plot.usgsandfid import GetUSGSIDandCorrFID

# subsetting
from .FIMsubset.xycoord import subsetFIM

# Fim visualization
from .vizualizationFIM import vizualizeFIM


# Statistics
from .statistics.calculatestatistics import CalculateStatistics

# For intersected HUC8 boundary
from .intersectedHUC import getIntersectedHUC8ID


# evaluation of FIM
from .fimevaluation.fims_setup import FIMService, fim_lookup
from .fimevaluation.run_fimeval import run_evaluation


#Enhancement using surrogate models [Importing those only if they are called]
def prepare_FORCINGs(*args, **kwargs):
    from .enhancement_withSM.preprocessFIM import prepare_FORCINGs as _impl
    return _impl(*args, **kwargs)

def predict_SM(*args, **kwargs):
    from .enhancement_withSM.SM_prediction import predict_SM as _impl
    return _impl(*args, **kwargs)

def get_building_exposure(*args, **kwargs):
    from .enhancement_withSM.building_exposure import get_building_exposure as _impl
    return _impl(*args, **kwargs)

def get_population_exposure(*args, **kwargs):
    from .enhancement_withSM.pop_exposure import get_population_exposure as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "DownloadHUC8",
    "getNWMRetrospectivedata",
    "runOWPHANDFIM",
    "getNWMForecasteddata",
    "getGEOGLOWSstreamflow",
    "plotNWMStreamflow",
    "getUSGSsitedata",
    "CompareNWMnUSGSStreamflow",
    "plotUSGSStreamflow",
    "plotSRC",
    "GetUSGSIDandCorrFID",
    "subsetFIM",
    "vizualizeFIM",
    "CalculateStatistics",
    "getIntersectedHUC8ID",
    "FIMService",
    "fim_lookup",
    "run_evaluation",
    "prepare_FORCINGs",
    "predict_SM",
    "get_building_exposure",
    "get_population_exposure",
]
