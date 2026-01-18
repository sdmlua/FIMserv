# Operational Enhanced FIM

## Overview

The **Operational Enhanced Flood Inundation Mapping (FIM)** framework provides a streamlined pipeline for real-time flood mapping and impact analysis. Its primary objective is to support emergency response operations by delivering timely, high-resolution flood information.


## Workflow

The core components of the framework are illustrated in the figure below:

![Workflow](graphics/operationalFIM.jpg)

## Key Capabilities

- **Real-time flood map generation** using surrogate models.
- **Automated processing pipeline** for rapid data acquisition, model execution, and visualization.
- **Scalable architecture** adaptable to different regions and data sources.

## Exposure Analysis

In addition to mapping flooded areas, the framework includes modules for:

- **Population exposure estimation**: Identifies the number of people affected by the flood event. For this it uses the population grids to analyze the exposed population. The framework provides the estimated exposed population and the spatial distribution of those population count using a unique way of vizualization.
![Workflow](graphics/population_exposure.png)

- **Building exposure analysis**: Detects flooded structures using geospatial building footprints and flood maps.
![Workflow](graphics/building_exposure.png)

These capabilities help quantify potential impacts and support decision-making in emergency situations.

## Application in Emergency Response

This framework is designed to:
- Provide first responders with up-to-date flood information
- Inform public warnings and evacuation strategies
- Aid in post-event damage assessments and recovery planning

## Usage
The integration of Surrogate Modeling (SM) framework with FIMserv can be run throgu Google Colab- as there GPU resources is available. 

SM enhancement usage Open In Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KD5tBJQD-q2K3xpylfqL_1OHtQrUi4GA)

## Reference
The trained Surrogate Model that is being applied for the enhancement of the FIMserv derieved Low-Fidelity results comes out of the following research, which is currently in review. 

**Preprint** :

**Supath Dhital**, Sagy Cohen, Parvaneh Nikrou, et al.  
Enhancement of low-fidelity flood inundation mapping through surrogate modeling. *ESS Open Archive*, November 03, 2025.  
[https://doi.org/10.22541/essoar.176218121.12875584/v1](https://doi.org/10.22541/essoar.176218121.12875584/v1)