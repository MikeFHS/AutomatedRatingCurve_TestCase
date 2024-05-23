# AutomatedRatingCurve_TestCase
Test case using the Automated Rating Curve (ARC) and FloodSpreader for the CIROH Workshop.

The Automated Rating Curve (ARC) is a python-based model that generates rating-curve information (depth, velocity, top-width) along streams.  ARC is typically used for large-scale applications (e.g., the Yellowstone BAsin), but can also be applied to smaller domains.  Inputs to ARC are peak- and base-flow data for each stream reach (often from continental-scale flow models) and geospatial data (land cover, DEM, stream shapefile).  ARC uses a simple approach to simulate a bathymetry profile within the DEM (https://doi.org/10.1016/j.jhydrol.2023.129769).  Using a volume-fill approach of Manning's Equation, mulitple flow rates are simulated.  Parameters (a and b) for a power-function are then generated where depth, velocity, and top-width are calculated as a function of flow (Q).  For example, Depth = a *Q^b.

The ARC-generated rating curves can then be used to create flood maps.  Here, we use ERDC's FloodSpreader model to show how flood maps can be generated when given a flow rate.

# Conda Environment:
    Typically AutoRoute and FloodSpreader are run using Anaconda Prompts (https://www.anaconda.com/download)
    To activate the "ARC" environment open an Anaconda Command Prompt and type "conda env create -f environment_ARC.yml"

# Model and Data Sources:
    floodspreader.exe is from ERDC: https://erdc-library.erdc.dren.mil/jspui/handle/11681/38783
    DEM data is from the National Elevation Dataset (1/3 arc second) - https://apps.nationalmap.gov/downloader/
    Land Cover data is from the National Land Cover Database 2011 - https://www.mrlc.gov/data/nlcd-2011-land-cover-conus
    Streamlines are from GeoGLoWS - http://geoglows-v2.s3-website-us-west-2.amazonaws.com/#streams/
    Return period flow data is from GeoGLoWS - http://geoglows-v2-retrospective.s3-website-us-west-2.amazonaws.com/#return-periods/

# Model/Data/Files are AS-IS with No Warranty:
Any scripts, files, etc. provided in this document are for testing purposes only and are provided "as is" and without any warranty of any kind.
