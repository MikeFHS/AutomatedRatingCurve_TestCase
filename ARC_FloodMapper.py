
#This code looks at a DEM raster to find the dimensions, then writes a script to create a STRM raster.

import sys
import os

try:
    import gdal 
    #import osr 
    #import ogr
    from gdalconst import GA_ReadOnly
except: 
    from osgeo import gdal
    #from osgeo import osr
    #from osgeo import ogr
    from osgeo.gdalconst import GA_ReadOnly

import subprocess
import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime

def convert_cell_size(dem_cell_size, dem_lower_left, dem_upper_right):
    """
    Determines the x and y cell sizes based on the geographic location

    Parameters
    ----------
    None. All input data is available in the parent object

    Returns
    -------
    None. All output data is set into the object

    """

    ### Get the cell size ###
    d_lat = np.fabs((dem_lower_left + dem_upper_right) / 2)

    ### Determine if conversion is needed
    if dem_cell_size > 0.5:
        # This indicates that the DEM is projected, so no need to convert from geographic into projected.
        x_cell_size = dem_cell_size
        y_cell_size = dem_cell_size
        projection_conversion_factor = 1

    else:
        # Reprojection from geographic coordinates is needed
        assert d_lat > 1e-16, "Please use lat and long values greater than or equal to 0."

        # Determine the latitude range for the model
        if d_lat >= 0 and d_lat <= 10:
            d_lat_up = 110.61
            d_lat_down = 110.57
            d_lon_up = 109.64
            d_lon_down = 111.32
            d_lat_base = 0.0

        elif d_lat > 10 and d_lat <= 20:
            d_lat_up = 110.7
            d_lat_down = 110.61
            d_lon_up = 104.64
            d_lon_down = 109.64
            d_lat_base = 10.0

        elif d_lat > 20 and d_lat <= 30:
            d_lat_up = 110.85
            d_lat_down = 110.7
            d_lon_up = 96.49
            d_lon_down = 104.65
            d_lat_base = 20.0

        elif d_lat > 30 and d_lat <= 40:
            d_lat_up = 111.03
            d_lat_down = 110.85
            d_lon_up = 85.39
            d_lon_down = 96.49
            d_lat_base = 30.0

        elif d_lat > 40 and d_lat <= 50:
            d_lat_up = 111.23
            d_lat_down = 111.03
            d_lon_up = 71.70
            d_lon_down = 85.39
            d_lat_base = 40.0

        elif d_lat > 50 and d_lat <= 60:
            d_lat_up = 111.41
            d_lat_down = 111.23
            d_lon_up = 55.80
            d_lon_down = 71.70
            d_lat_base = 50.0

        elif d_lat > 60 and d_lat <= 70:
            d_lat_up = 111.56
            d_lat_down = 111.41
            d_lon_up = 38.19
            d_lon_down = 55.80
            d_lat_base = 60.0

        elif d_lat > 70 and d_lat <= 80:
            d_lat_up = 111.66
            d_lat_down = 111.56
            d_lon_up = 19.39
            d_lon_down = 38.19
            d_lat_base = 70.0

        elif d_lat > 80 and d_lat <= 90:
            d_lat_up = 111.69
            d_lat_down = 111.66
            d_lon_up = 0.0
            d_lon_down = 19.39
            d_lat_base = 80.0

        else:
            raise AttributeError('Please use legitimate (0-90) lat and long values.')

        ## Convert the latitude ##
        d_lat_conv = d_lat_down + (d_lat_up - d_lat_down) * (d_lat - d_lat_base) / 10
        y_cell_size = dem_cell_size * d_lat_conv * 1000.0  # Converts from degrees to m

        ## Longitude Conversion ##
        d_lon_conv = d_lon_down + (d_lon_up - d_lon_down) * (d_lat - d_lat_base) / 10
        x_cell_size = dem_cell_size * d_lon_conv * 1000.0  # Converts from degrees to m

        ## Make sure the values are in bounds ##
        if d_lat_conv < d_lat_down or d_lat_conv > d_lat_up or d_lon_conv < d_lon_up or d_lon_conv > d_lon_down:
            raise ArithmeticError("Problem in conversion from geographic to projected coordinates")

        ## Calculate the conversion factor ##
        projection_conversion_factor = 1000.0 * (d_lat_conv + d_lon_conv) / 2.0
    return x_cell_size, y_cell_size, projection_conversion_factor

def FindFlowRateForEachCOMID_Ensemble(comid_file_lines, flow_event_num, COMID_to_ID, MinCOMID, COMID_Unique_Flow):    
    num_lines = len(comid_file_lines)
    for n in range(1,num_lines):
        splitlines = comid_file_lines[n].strip().split(',')
        COMID = splitlines[0]
        Q = splitlines[1:]
        i = COMID_to_ID[int(COMID)-MinCOMID]
        COMID_Unique_Flow[i] = float(Q[flow_event_num])
    return

def Calculate_TW_D_ForEachCOMID(CurveParamFileName, COMID_Unique_Flow, COMID_Unique, COMID_to_ID, MinCOMID, Q_Fraction, T_Rast, W_Rast, TW_MultFact):
    num_unique = len(COMID_Unique)
    COMID_Unique_TW = np.zeros(num_unique)
    COMID_Unique_Depth = np.zeros(num_unique)
    COMID_NumRecord = np.zeros(num_unique)
    print('\nOpening and Reading ' + CurveParamFileName)
    infile = open(CurveParamFileName,'r')
    lines = infile.readlines()
    infile.close()
    
    Dbv = 0.0
    QB = 0.0
    Vbv = 0.0
    Tbv = 0.0
    
    num_lines = len(lines)
    for n in range(1,num_lines):
        (COMID, R, C, E, DEM_E, QMax, Da, Db, Ta, Tb , Va, Vb ) = lines[n].strip().split(',')
        E = DEM_E
        i = COMID_to_ID[int(COMID)-MinCOMID]
        if float(Da)>0.0 and float(Db)>0.0 and float(Ta)>0.0 and float(Tb)>0.0:
            COMID_NumRecord[i] = COMID_NumRecord[i] + 1
            Q = COMID_Unique_Flow[i] * Q_Fraction
            Depth = float(Dbv) + float(Da) * pow(Q,float(Db))
            TopWidth = float(Tbv) + float(Ta) * pow(Q,float(Tb))
            TopWidth = TopWidth * TW_MultFact
            
            T_Rast[int(R),int(C)] = TopWidth
            W_Rast[int(R),int(C)] = Depth + float(E)
            
            #Calculate the Average Depth and TopWidth
            COMID_Unique_TW[i] = ( COMID_Unique_TW[i]*(COMID_NumRecord[i]-1) + TopWidth ) / COMID_NumRecord[i]
            COMID_Unique_Depth[i] = ( COMID_Unique_Depth[i]*(COMID_NumRecord[i]-1) + Depth ) / COMID_NumRecord[i]
            
    TopWidthMax = COMID_Unique_TW.max()
    return (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax)

def Calculate_TW_D_ForEachCOMID_VDTDatabase(VDTDatabaseFileName, COMID_Unique_Flow, COMID_Unique, COMID_to_ID, MinCOMID, Q_Fraction, T_Rast, W_Rast, TW_MultFact):
    num_unique = len(COMID_Unique)
    COMID_Unique_TW = np.zeros(num_unique)
    COMID_Unique_Depth = np.zeros(num_unique)
    COMID_NumRecord = np.zeros(num_unique)
    print('\nOpening and Reading ' + VDTDatabaseFileName)
    infile = open(VDTDatabaseFileName,'r')
    lines = infile.readlines()
    infile.close()
    
    num_lines = len(lines)
    for n in range(1,num_lines):
        linesplit = lines[n].strip().split(',')
        (COMID, R, C, E, QB) = linesplit[0:5]
        QVTW = linesplit[5:]
        num_q = int(len(QVTW)/4)
        i = COMID_to_ID[int(COMID)-MinCOMID]
        
        #print(str(COMID_Unique_Flow[i]) + '  ' + '  ' + str(QB) + '  ' + str(QVTW[0]) + '   ' + str(QVTW[(num_q-1)*4]))
        
        Depth = -1.0 
        TopWidth = -1.0 
        WSE = -1.0
        
        #Convert from String to floating point
        QVTW = np.array(QVTW, dtype=np.float32)
        R = int(R)
        C = int(C)
        E = float(E)
        QB = float(QB)
        
        
        if COMID_Unique_Flow[i] <= QB:   #Flow is below baseflow, so ignore
            TopWidth = 1.0
            Depth = 0.001
            WSE = E
        elif COMID_Unique_Flow[i] >=QVTW[0]:
            TopWidth = QVTW[2]
            WSE = QVTW[3]
            Depth = WSE - E
        elif COMID_Unique_Flow[i] <= QVTW[(num_q-1)*4]:
            TopWidth = QVTW[(num_q-1)*4+2]
            WSE = QVTW[(num_q-1)*4+3]
            Depth = WSE - E
        else:
            for x in range(1,num_q):
                if COMID_Unique_Flow[i] >= QVTW[x*4]:
                    denom_val = (QVTW[(x-1)*4] - QVTW[x*4])
                    if abs(denom_val)>0.0001:
                        fractval = (COMID_Unique_Flow[i] - QVTW[x*4]) / denom_val
                        WSE = QVTW[x*4+3] + fractval * (QVTW[(x-1)*4+3] - QVTW[(x-1)*4+3])
                        if WSE < E:
                            WSE = E
                        Depth = WSE - E
                        TopWidth = QVTW[x*4+2] + fractval * (QVTW[(x-1)*4+2] - QVTW[(x-1)*4+2])
                    else:
                        WSE = QVTW[x*4+3]
                        if WSE < E:
                            WSE = E
                        Depth = WSE - E
                        TopWidth = QVTW[x*4+2]
                    #TopWidth = TopWidth * TW_MultFact
                    break
        TopWidth = TopWidth * TW_MultFact
        #print(str(Depth) + '  ' + str(TopWidth))
        if TopWidth>0.0001 and WSE>0.0001:
            T_Rast[R,C] = TopWidth
            W_Rast[R,C] = WSE
        
        
        #Calculate the Average Depth and TopWidth
        if Depth > 0.00001 and TopWidth > 0.0001:
            COMID_NumRecord[i] = COMID_NumRecord[i] + 1
            COMID_Unique_TW[i] = ( COMID_Unique_TW[i]*(COMID_NumRecord[i]-1) + TopWidth ) / COMID_NumRecord[i]
            COMID_Unique_Depth[i] = ( COMID_Unique_Depth[i]*(COMID_NumRecord[i]-1) + Depth ) / COMID_NumRecord[i]
        
        #if int(COMID) == 750189551:
        #    print('Q =    ' + str(COMID_Unique_Flow[i]))
        #    print('Depth = ' + str(Depth))
        #    print('WSE = ' + str(WSE)) 
        #    print('TW = ' + str(TopWidth))
        
        #T_Rast[int(R),int(C)] = 100.1
        #W_Rast[int(R),int(C)] = float(E) + 1.1
        
        #print(COMID)
        #print(R)
        #print(C)
        #print(COMID_Unique_Flow[i])
        #print(Depth)
        #print(TopWidth)
    
    TopWidthMax = COMID_Unique_TW.max()
    return (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax)

    

def Get_Raster_Details(DEM_File):
    print(DEM_File)
    gdal.Open(DEM_File, gdal.GA_ReadOnly)
    data = gdal.Open(DEM_File)
    geoTransform = data.GetGeoTransform()
    ncols = int(data.RasterXSize)
    nrows = int(data.RasterYSize)
    minx = geoTransform[0]
    dx = geoTransform[1]
    maxy = geoTransform[3]
    dy = geoTransform[5]
    maxx = minx + dx * ncols
    miny = maxy + dy * nrows
    Rast_Projection = data.GetProjectionRef()
    data = None
    return minx, miny, maxx, maxy, dx, dy, ncols, nrows, geoTransform, Rast_Projection

def Read_Raster_GDAL(InRAST_Name):
    try:
        dataset = gdal.Open(InRAST_Name, gdal.GA_ReadOnly)     
    except RuntimeError:
        sys.exit(" ERROR: Field Raster File cannot be read!")
    # Retrieve dimensions of cell size and cell count then close DEM dataset
    geotransform = dataset.GetGeoTransform()
    # Continue grabbing geospatial information for this use...
    band = dataset.GetRasterBand(1)
    RastArray = band.ReadAsArray()
    #global ncols, nrows, cellsize, yll, yur, xll, xur
    ncols=band.XSize
    nrows=band.YSize
    band = None
    cellsize = geotransform[1]
    yll = geotransform[3] - nrows * np.fabs(geotransform[5])
    yur = geotransform[3]
    xll = geotransform[0];
    xur = xll + (ncols)*geotransform[1]
    lat = np.fabs((yll+yur)/2.0)
    Rast_Projection = dataset.GetProjectionRef()
    dataset = None
    print('Spatial Data for Raster File:')
    print('   ncols = ' + str(ncols))
    print('   nrows = ' + str(nrows))
    print('   cellsize = ' + str(cellsize))
    print('   yll = ' + str(yll))
    print('   yur = ' + str(yur))
    print('   xll = ' + str(xll))
    print('   xur = ' + str(xur))
    return RastArray, ncols, nrows, cellsize, yll, yur, xll, xur, lat, geotransform, Rast_Projection


def Write_Output_Raster(s_output_filename, raster_data, ncols, nrows, dem_geotransform, dem_projection, s_file_format, s_output_type):   
    o_driver = gdal.GetDriverByName(s_file_format)  #Typically will be a GeoTIFF "GTiff"
    #o_metadata = o_driver.GetMetadata()
    
    # Construct the file with the appropriate data shape
    o_output_file = o_driver.Create(s_output_filename, xsize=ncols, ysize=nrows, bands=1, eType=s_output_type)
    
    # Set the geotransform
    o_output_file.SetGeoTransform(dem_geotransform)
    
    # Set the spatial reference
    o_output_file.SetProjection(dem_projection)
    
    # Write the data to the file
    o_output_file.GetRasterBand(1).WriteArray(raster_data)
    
    # Once we're done, close properly the dataset
    o_output_file = None

def CreateSimpleFloodMap(RR, CC, T_Rast, W_Rast, E, B, nrows, ncols, sd, TW_m, dx, dy, LocalFloodOption, COMID_Unique, COMID_to_ID, MinCOMID, COMID_Unique_TW, COMID_Unique_Depth, TW, TW_MultFact):
    #Flood = np.zeros((nrows+2,ncols+2))
    
    COMID_Averaging_Method = 0
    
    Flooded = np.zeros((nrows+2,ncols+2)).astype(float)
    
    #Now go through each cell
    num_nonzero = len(RR)
    for i in range(num_nonzero):
        r = RR[i]
        c = CC[i]
        r_use = r
        c_use = c
        E_Min = E[r,c]
        
        #Now start with rows and start flooding everything in site
        if COMID_Averaging_Method!=0 or W_Rast[r-1,c-1]<0.001 or T_Rast[r-1,c-1]<0.00001:
            #Get COMID, TopWidth, and Depth Information for this cell
            COMID_Value = int(B[r,c])
            iii = COMID_to_ID[COMID_Value - MinCOMID]
            COMID_TW_m = COMID_Unique_TW[iii]
            COMID_D = COMID_Unique_Depth[iii]
            WSE = float(E[r_use,c_use] + COMID_D)
        else:
            #These are Based on the AutoRoute Results, not averaged for COMID
            WSE = W_Rast[r-1,c-1]  #Have to have the '-1' because of the Row and Col being inset on the B raster.
            COMID_TW_m = T_Rast[r-1,c-1]
            
        if WSE<0.001 or COMID_TW_m<0.00001:
            continue
        
        if COMID_TW_m > TW_m:
            COMID_TW_m = TW_m
        COMID_TW = int(max(round(COMID_TW_m/dx,0),round(COMID_TW_m/dy,0)))  #This is how many cells we will be looking at surrounding our stream cell
        if COMID_TW<=1:
            COMID_TW=2 
        
        
        
        r_min = r_use-COMID_TW
        r_max = r_use+COMID_TW+1
        if r_min<1:
            r_min = 1 
        if r_max>(nrows+1):
            r_max=nrows+1
        c_min = c_use-COMID_TW
        c_max = c_use+COMID_TW+1
        if c_min<1:
            c_min = 1 
        if c_max>(ncols+1):
            c_max=ncols+1
        
        for rrr in range(r_min,r_max):
            for ccc in range(c_min,c_max):
                if WSE>E[rrr,ccc]:
                    Flooded[rrr,ccc] = 1
    return Flooded

def Calculate_Depth_TopWidth_TWMax(CurveParamFileName, VDTDatabaseFileName, COMID_Unique_Flow, COMID_Unique, COMID_to_ID, MinCOMID, Q_Fraction, T_Rast, W_Rast, TW_MultFact, TopWidthPlausibleLimit, dx, dy):
    if len(CurveParamFileName)>1:
        (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax) = Calculate_TW_D_ForEachCOMID(CurveParamFileName, COMID_Unique_Flow, COMID_Unique, COMID_to_ID, MinCOMID, Q_Fraction, T_Rast, W_Rast, TW_MultFact)
    elif len(VDTDatabaseFileName)>1:
        (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax) = Calculate_TW_D_ForEachCOMID_VDTDatabase(VDTDatabaseFileName, COMID_Unique_Flow, COMID_Unique, COMID_to_ID, MinCOMID, Q_Fraction, T_Rast, W_Rast, TW_MultFact)
    print('Maximum Top Width = ' + str(TopWidthMax))
    
    for x in range(len(COMID_Unique)):
        if COMID_Unique_TW[x]>TopWidthPlausibleLimit:
            print('Ignoring ' + str(COMID_Unique[x]) + '  ' + str(COMID_Unique_Flow[x])  + '  ' + str(COMID_Unique_Flow[x]*Q_Fraction) + '  ' + str(COMID_Unique_Depth[x]) + '  ' + str(COMID_Unique_TW[x]))             
    
    if TopWidthPlausibleLimit < TopWidthMax:
        TopWidthMax = TopWidthPlausibleLimit
    
    #Create a Weight Box and an Elipse Mask that can be used for all of the cells
    X_cells = round(TopWidthMax/dx,0)
    Y_cells = round(TopWidthMax/dy,0)
    TW = int(max(Y_cells,X_cells))  #This is how many cells we will be looking at surrounding our stream cell
    
    return COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, TW

def Cheap_FloodMapping_MainFunction(S, DEM, E, B, RR, CC, nrows, ncols, dx, dy, COMID_Unique, num_comids, MinCOMID, MaxCOMID, COMID_to_ID, COMID_Unique_Flow, CurveParamFileName, VDTDatabaseFileName, Q_Fraction, TopWidthPlausibleLimit, TW_MultFact, LocalFloodOption):
    
    #These are gridded data from Curve Parameter or VDT Database File
    T_Rast = np.zeros((nrows,ncols))
    W_Rast = np.zeros((nrows,ncols))
    T_Rast = T_Rast - 1.0
    W_Rast = W_Rast - 1.0
    
    
    #Calculate an Average Top Width and Depth for each stream reach.
    (COMID_Unique_TW, COMID_Unique_Depth, TopWidthMax, TW) = Calculate_Depth_TopWidth_TWMax(CurveParamFileName, VDTDatabaseFileName, COMID_Unique_Flow, COMID_Unique, COMID_to_ID, MinCOMID, Q_Fraction, T_Rast, W_Rast, TW_MultFact, TopWidthPlausibleLimit, dx, dy)
    
    #(WeightBox, ElipseMask) = CreateWeightAndElipseMask(TW, dx, dy, TW_MultFact)  #3D Array with the same row/col dimensions as the WeightBox
    
    
    #Create a simple Flood Map Data
    search_dist_for_min_elev = 0
    print('Creating Rough Flood Map Data...')
    Flood = CreateSimpleFloodMap(RR, CC, T_Rast, W_Rast, E, B, nrows, ncols, search_dist_for_min_elev, TopWidthMax, dx, dy, LocalFloodOption, COMID_Unique, COMID_to_ID, MinCOMID, COMID_Unique_TW, COMID_Unique_Depth, TW, TW_MultFact)
    
    return Flood[1:nrows+1,1:ncols+1]


def ArcFloodMapper(DEM_File, STRM_File, FlowFileName, CurveParamFileName, VDTDatabaseFileName, Flood_File, FloodImpact_File, Q_Fraction, TopWidthPlausibleLimit, TW_MultFact, LocalFloodOption):
    
    print('Get the Raster Dimensions for ' + DEM_File)
    (minx, miny, maxx, maxy, dx, dy, ncols, nrows, dem_geoTransform, dem_projection) = Get_Raster_Details(DEM_File)
    cellsize_x = abs(float(dx))
    cellsize_y = abs(float(dy))
    lat_base = float(maxy) - 0.5*cellsize_y
    lon_base = float(minx) + 0.5*cellsize_x
    
    print('Opening ' + STRM_File)
    (S, ncols, nrows, cellsize, yll, yur, xll, xur, lat, s_geotransform, s_projection) = Read_Raster_GDAL(STRM_File)
        
    print('Opening ' + DEM_File)
    (DEM, ncols, nrows, cellsize, yll, yur, xll, xur, lat, dem_geotransform, dem_projection) = Read_Raster_GDAL(DEM_File)
    
    cellsize_x = abs(float(dx))
    cellsize_y = abs(float(dy))
    lat_base = float(maxy) - 0.5*cellsize_y
    lon_base = float(minx) + 0.5*cellsize_x
        
        
    E = np.zeros((nrows+2,ncols+2))  #Create an array that is slightly larger than the STRM Raster Array
    E[1:(nrows+1), 1:(ncols+1)] = DEM
    E = E.astype(float)
    
    #Get Cellsize Information
    (dx, dy, dm) = convert_cell_size(cellsize, yll, yur)
    dz = pow(dx*dx+dy*dy,0.5)
    
    #Get list of Unique Stream IDs.  Also find where all the cell values are.
    B = np.zeros((nrows+2,ncols+2))  #Create an array that is slightly larger than the STRM Raster Array
    B[1:(nrows+1), 1:(ncols+1)] = S
    B = B.astype(int)
    (RR,CC) = B.nonzero()
    COMID_Unique = np.unique(B)
    COMID_Unique = np.delete(COMID_Unique, 0)  #We don't need the first entry of zero
    #Sort from Smallest to highest values
    COMID_Unique = np.sort(COMID_Unique).astype(int)
    num_comids = len(COMID_Unique)
    
    #print(COMID_Unique)
    
    MinCOMID = int(COMID_Unique.min())
    MaxCOMID = int(COMID_Unique.max())
    print('\nCOMID Ranges from ' + str(MinCOMID) + ' to ' + str(MaxCOMID))
    
    COMID_to_ID = np.zeros(MaxCOMID-MinCOMID+1).astype(int)
    COMID_to_ID = COMID_to_ID - 1
    
    #Get the Unique Identifier Set.  The COMID_to_ID values will now be -1 for nondata, and the rest range from 0 to num_comids-1
    for i in range(num_comids):
        COMID_to_ID[int(COMID_Unique[i]-MinCOMID)] = i
    
    
    #COMID Flow File Read-in
    num_unique = len(COMID_Unique)
    COMID_Unique_Flow = np.zeros(num_unique)
    print('\nOpening and Reading ' + FlowFileName)
    infile = open(FlowFileName,'r')
    comid_file_lines = infile.readlines()
    infile.close()
    
    #Order from highest to lowest flow
    ls = comid_file_lines[1].split(',')
    num_flows = len(ls)-1
    print('Evaluating ' + str(num_flows) + ' Flow Events')
    
    
    #Creating the Weight and Eclipse Boxes
    print('Creating the Weight and Eclipse Boxes')
    TW = int( max( round(TopWidthPlausibleLimit/dx,0), round(TopWidthPlausibleLimit/dy,0) ) )  #This is how many cells we will be looking at surrounding our stream cell
    
    Flood_Ensemble = np.zeros((nrows,ncols))
    
    #Go through all the Flow Events
    for flow_event_num in range(num_flows):
        print('Working on Flow Event ' + str(flow_event_num))
        #Get an Average Flow rate associated with each stream reach.
        FindFlowRateForEachCOMID_Ensemble(comid_file_lines, flow_event_num, COMID_to_ID, MinCOMID, COMID_Unique_Flow)
        Flood = Cheap_FloodMapping_MainFunction(S, DEM, E, B, RR, CC, nrows, ncols, dx, dy, COMID_Unique, num_comids, MinCOMID, MaxCOMID, COMID_to_ID, COMID_Unique_Flow, CurveParamFileName, VDTDatabaseFileName, Q_Fraction, TopWidthPlausibleLimit, TW_MultFact, LocalFloodOption)
        Flood_Ensemble = Flood_Ensemble + Flood
    
    #Turn into a percentage
    Flood_Ensemble = (100 * Flood_Ensemble / num_flows).astype(int)
    
    print('Creating Ensemble Flood Map...' + str(Flood_File))
    Write_Output_Raster(Flood_File, Flood_Ensemble, ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Int32)
    
    return


if __name__ == "__main__":
    
    
    #These are the main inputs to the model
    
    MainFolder = ''
    
    LocalFloodOption = False
    Q_Fraction = 1.0
    TopWidthPlausibleLimit = 250.0
    TW_MultFact = 1.5
    DEM_File = MainFolder + 'DEM/Gardiner_DEM.tif' 
    STRM_File = MainFolder + 'STRM/Gardiner_STRM_Raster_Clean.tif' 
    Flood_File = MainFolder + 'FloodMap/Gardiner_FloodMap.tif'
    FloodImpact_File = '' 
    FlowFileName = MainFolder + 'FlowFile/COMID_Q_qout_max.txt'
    
    VDTDatabaseFileName = ''
    CurveParamFileName = ''
    
    
    #Option to input a Curve Paramater File, or the VDT_Database File
    CurveParamFileName = MainFolder + 'VDT/Gardiner_CurveFile.csv'
    #VDTDatabaseFileName = 'VDT/Gardiner_VDT_Database.txt'
    
    Model_Start_Time = datetime.now()
    ArcFloodMapper(DEM_File, STRM_File, FlowFileName, CurveParamFileName, VDTDatabaseFileName, Flood_File, FloodImpact_File, Q_Fraction, TopWidthPlausibleLimit, TW_MultFact, LocalFloodOption)
    Model_Simulation_Time = datetime.now() - Model_Start_Time
    print('\n' + 'Simulation time (sec)= ' + str(Model_Simulation_Time.seconds))
    
    