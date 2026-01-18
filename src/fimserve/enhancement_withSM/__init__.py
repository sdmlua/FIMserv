#Enhancement using surrogate models
def prepare_FORCINGs(*args, **kwargs):
    from .preprocessFIM import prepare_FORCINGs as _impl
    return _impl(*args, **kwargs)

def enhanceFIM(*args, **kwargs):
    from .SM_prediction import enhanceFIM as _impl
    return _impl(*args, **kwargs)

def getbuilding_exposure(*args, **kwargs):
    from .building_exposure import getbuilding_exposure as _impl
    return _impl(*args, **kwargs)

def getpopulation_exposure(*args, **kwargs):
    from .pop_exposure import getpopulation_exposure as _impl
    return _impl(*args, **kwargs)

__all__ = [
    "prepare_FORCINGs",
    "enhanceFIM",
    "getbuilding_exposure",
    "getpopulation_exposure",
]