
import sys
import os

try:
    import gdal 
    #import osr 
    #import ogr
    #from gdalconst import GA_ReadOnly
except: 
    from osgeo import gdal
    #from osgeo import osr
    #from osgeo import ogr
    #from osgeo.gdalconst import GA_ReadOnly

import numpy as np
import netCDF4   #conda install netCDF4


def Process_AutoRoute_Geospatial_Data():
    
    #Input Dataset
    Main_Directory = ''
    ARC_Folder = 'ARC_InputFiles'
    ARC_FileName = os.path.join(ARC_Folder, 'Gardiner_ARC_Input_File.txt')
    DEM_File = 'DEM/Gardiner_DEM.tif'
    LandCoverFile = 'LandCover/NLCD_2011_EPSG4269.tif'
    StrmSHP = 'StrmShp/Gardiner_GeoGLoWS_StreamShapefile.shp'
    FlowNC = 'FlowData/returnperiods_714.nc'
    VDT_Test_File = 'VDT/Gardiner_VDT_FS.csv'
    
    #Datasets to be Created
    STRM_File = 'STRM/Gardiner_STRM_Raster.tif'
    STRM_File_Clean = STRM_File.replace('.tif','_Clean.tif')
    LAND_File = 'LAND/Gardiner_LAND_Raster.tif'
    FLOW_File = 'FLOW/Gardiner_FlowFile.txt'
    FlowFileFolder = 'FlowFile'
    BathyFileFolder = 'Bathymetry'
    FloodFolder = 'FloodMap'
    ARC_Folder = 'ARC_InputFiles'
    ManningN = 'LAND/AR_Manning_n_for_NLCD_MED.txt'
    VDT_File = 'VDT/Gardiner_VDT_Database.txt'
    Curve_File = 'VDT/Gardiner_CurveFile.csv'
    FloodMapFile = FloodFolder + '/' + 'Gardiner_ARC_Flood.tif'
    DepthMapFile = FloodFolder + '/' + 'Gardiner_ARC_Depth.tif'
    ARC_BathyFile = BathyFileFolder + '/' + 'Gardiner_ARC_Bathy.tif'
    
    #Create Folders
    Create_Folder('STRM')
    Create_Folder('LAND')
    Create_Folder('FLOW')
    Create_Folder('VDT')
    Create_Folder(FloodFolder)
    Create_Folder(FlowFileFolder)
    Create_Folder(ARC_Folder)
    Create_Folder(BathyFileFolder)
    
    
    
    #Get the Spatial Information from the DEM Raster
    (minx, miny, maxx, maxy, dx, dy, ncols, nrows, dem_geoTransform, dem_projection) = Get_Raster_Details(DEM_File)
    projWin_extents = [minx, maxy, maxx, miny]
    outputBounds = [minx, miny, maxx, maxy]  #https://gdal.org/api/python/osgeo.gdal.html
    
    
    #Create Land Dataset
    if os.path.isfile(LAND_File):
        print(LAND_File + ' Already Exists')
    else: 
        print('Creating ' + LAND_File) 
        Create_AR_LandRaster(LandCoverFile, LAND_File, projWin_extents, ncols, nrows)
    
    #Create Stream Raster
    if os.path.isfile(STRM_File):
        print(STRM_File + ' Already Exists')
    else:
        print('Creating ' + STRM_File)
        Create_AR_StrmRaster(StrmSHP, STRM_File, outputBounds, minx, miny, maxx, maxy, dx, dy, ncols, nrows, 'LINKNO')
    
    #Clean Stream Raster
    if os.path.isfile(STRM_File_Clean):
        print(STRM_File_Clean + ' Already Exists')
    else:
        print('Creating ' + STRM_File_Clean)
        Clean_STRM_Raster(STRM_File, STRM_File_Clean)
    
    
    
    
    #Read all of the recurrence interval flow information for each stream reach
    id_index = 'rivid'
    (ID, QMax, Q2, Q5, Q10, Q25, Q50, Q100) = PullNetCDFInfo(FlowNC, id_index, 'qout_max', 'rp2', 'rp5', 'rp10', 'rp25', 'rp50', 'rp100')
    ID_np = np.asarray(ID).astype(int)
    num_rivid = len(ID_np)
    print(num_rivid)
    
    #Get the unique values for all the stream ids
    (S, ncols, nrows, cellsize, yll, yur, xll, xur, lat, dem_geotransform, dem_projection) = Read_Raster_GDAL(STRM_File_Clean)
    (RR,CC) = S.nonzero()
    num_strm_cells = len(RR)
    COMID_Unique = np.unique(S)
    COMID_Unique = np.delete(COMID_Unique, 0)  #We don't need the first entry of zero
    COMID_Unique = np.sort(COMID_Unique).astype(int)
    num_comids = len(COMID_Unique)
    
    #Organize the recurrence interval flow data so it easily links with stream raster data
    print('Linking data from ' + STRM_File_Clean + '  with  ' + FlowNC)
    print('\n\nCreating COMID Flow Files in the folder ' + FlowFileFolder)
    MinCOMID = int(COMID_Unique.min())
    MaxCOMID = int(COMID_Unique.max())
    Create_COMID_Flow_Files(COMID_Unique, num_comids, MinCOMID, MaxCOMID, FlowFileFolder, ID, QMax, Q2, Q5, Q10, Q25, Q50, Q100)
    
    #Write the Flow File for ARC
    num_unique_comid = len(COMID_Unique)
    print('Writing ' + FLOW_File + ' for ' + str(num_unique_comid) + ' unique ID values')
    out_file = open(FLOW_File,'w')
    out_str = 'COMID,qout_max,rp2,rp5,rp10,rp25,rp50,rp100'
    out_file.write(out_str)
    for x in range(num_unique_comid):
        out_str = '\n' + str(COMID_Unique[x]) + ',' + str(QMax[x]) + ',' + str(Q2[x]) + ',' + str(Q5[x]) + ',' + str(Q10[x]) + ',' + str(Q25[x]) + ',' + str(Q50[x]) + ',' + str(Q100[x])
        out_file.write(out_str)
    out_file.close()
    
    #Create a Baseline Manning N File
    print('Creating Manning n file: ' + ManningN)
    Create_BaseLine_Manning_n_File(ManningN)
    
    #Create a Starting AutoRoute Input File
    ARC_FileName = os.path.join(ARC_Folder,'ARC_Input_File.txt')
    print('Creating ARC Input File: ' + ARC_FileName)
    COMID_Q_File = FlowFileFolder + '/' + 'COMID_Q_qout_max.txt'
    Create_ARC_Model_Input_File(Main_Directory, ARC_FileName, DEM_File, COMID_Q_File, 'COMID', 'qout_max', 'rp2', STRM_File_Clean, LAND_File, FLOW_File, VDT_File, Curve_File, ManningN, FloodMapFile, DepthMapFile, ARC_BathyFile, VDT_Test_File)
    
    
    print('\n\n')
    print('Next Step is to Run Automated_Rating_Curve_Generator.py by copying the following into the Command Prompt:')
    print('python Automated_Rating_Curve_Generator.py ' + Main_Directory + ARC_Folder + '/' + 'ARC_Input_File.txt')
    
    return

def Create_Folder(F):
    if not os.path.exists(F): 
        os.makedirs(F)
    return

def Create_ARC_Model_Input_File(MD, ARC_Input_File, DEM_File, COMID_Q_File, COMID_Param, Q_Param, Q_BF_Param, STRM_File_Clean, LAND_File, FLOW_File, VDT_File, Curve_File, ManningN, FloodMapFile, DepthMapFile, ARC_BathyFile, VDT_Test_File):
    out_file = open(os.path.join(MD,ARC_Input_File),'w')
    out_file.write('#ARC_Inputs')
    out_file.write('\n' + 'DEM_File	' + os.path.join(MD,DEM_File))
    out_file.write('\n' + 'Stream_File	' + os.path.join(MD,STRM_File_Clean))
    out_file.write('\n' + 'LU_Raster_SameRes	' + os.path.join(MD,LAND_File))
    out_file.write('\n' + 'LU_Manning_n	' + os.path.join(MD,ManningN))
    out_file.write('\n' + 'Flow_File	' + os.path.join(MD,FLOW_File))
    out_file.write('\n' + 'Flow_File_ID	' + COMID_Param)
    out_file.write('\n' + 'Flow_File_BF	' + Q_BF_Param)
    out_file.write('\n' + 'Flow_File_QMax	' + Q_Param)
    out_file.write('\n' + 'Spatial_Units	deg')
    out_file.write('\n' + 'X_Section_Dist	5000.0')
    out_file.write('\n' + 'Degree_Manip	6.1')
    out_file.write('\n' + 'Degree_Interval	1.5')
    out_file.write('\n' + 'Low_Spot_Range	10')
    out_file.write('\n' + 'Str_Limit_Val	1')
    out_file.write('\n' + 'Gen_Dir_Dist	10')
    out_file.write('\n' + 'Gen_Slope_Dist	10')
    
    out_file.write('\n\n#VDT_Output_File_and_CurveFile')
    out_file.write('\n' + 'Print_VDT_Database	' + os.path.join(MD,VDT_File))
    out_file.write('\n' + 'Print_Curve_File	' + os.path.join(MD,Curve_File))
    
    out_file.write('\n\n#VDT_File_For_TestingPurposes_Only')
    out_file.write('\n' + 'Print_VDT	' + os.path.join(MD,VDT_Test_File))
    out_file.write('\n' + 'OutFLD	' + os.path.join(MD,FloodMapFile))
    
    out_file.write('\n\n#Bathymetry_Information')
    out_file.write('\n' + 'Bathy_Trap_H	0.20')
    out_file.write('\n' + 'AROutBATHY	' + ARC_BathyFile)
    out_file.close()
    

def Create_BaseLine_Manning_n_File(ManningN):
    out_file = open(ManningN,'w')
    out_file.write('LC_ID	Description	Manning_n')
    out_file.write('\n' + '11	Water	0.030')
    out_file.write('\n' + '21	Dev_Open_Space	0.013')
    out_file.write('\n' + '22	Dev_Low_Intesity	0.050')
    out_file.write('\n' + '23	Dev_Med_Intensity	0.075')
    out_file.write('\n' + '24	Dev_High_Intensity	0.100')
    out_file.write('\n' + '31	Barren_Land	0.030')
    out_file.write('\n' + '41	Decid_Forest	0.120')
    out_file.write('\n' + '42	Evergreen_Forest	0.120')
    out_file.write('\n' + '43	Mixed_Forest	0.120')
    out_file.write('\n' + '52	Shrub	0.050')
    out_file.write('\n' + '71	Grass_Herb	0.030')
    out_file.write('\n' + '81	Pasture_Hay	0.040')
    out_file.write('\n' + '82	Cultivated_Crops	0.035')
    out_file.write('\n' + '90	Woody_Wetlands	0.100')
    out_file.write('\n' + '95	Emergent_Herb_Wet	0.100')
    out_file.close()

def Create_COMID_Flow_Files(COMID_Unique, num_comids, MinCOMID, MaxCOMID, FlowFileFolder, ID, QMax, Q2, Q5, Q10, Q25, Q50, Q100):
    fmax = open(str(FlowFileFolder) + '/COMID_Q_qout_max.txt', 'w')
    fmax.write('COMID,qout_max')
    f2 = open(str(FlowFileFolder) + '/COMID_Q_rp2.txt', 'w')
    f2.write('COMID,rp2')
    f5 = open(str(FlowFileFolder) + '/COMID_Q_rp5.txt', 'w')
    f5.write('COMID,rp5')
    f10 = open(str(FlowFileFolder) + '/COMID_Q_rp10.txt', 'w')
    f10.write('COMID,rp10')
    f25 = open(str(FlowFileFolder) + '/COMID_Q_rp25.txt', 'w')
    f25.write('COMID,rp25')
    f50 = open(str(FlowFileFolder) + '/COMID_Q_rp50.txt', 'w')
    f50.write('COMID,rp50')
    f100 = open(str(FlowFileFolder) + '/COMID_Q_rp100.txt', 'w')
    f100.write('COMID,rp100')
    for i in range(num_comids):
        COMID = COMID_Unique[i]
        x = np.where(ID==COMID)
        x = int(x[0])
        
        fmax.write('\n' + str(COMID) + ',' + str(QMax[x]))
        f2.write('\n' + str(COMID) + ',' + str(Q2[x]))
        f5.write('\n' + str(COMID) + ',' + str(Q5[x]))
        f10.write('\n' + str(COMID) + ',' + str(Q10[x]))
        f25.write('\n' + str(COMID) + ',' + str(Q25[x]))
        f50.write('\n' + str(COMID) + ',' + str(Q50[x]))
        f100.write('\n' + str(COMID) + ',' + str(Q100[x]))
    fmax.close()
    f2.close()
    f5.close()
    f10.close()
    f25.close()
    f50.close()
    f100.close()
    return


def PullNetCDFInfo(infilename, id_index, q_max, q_2, q_5, q_10, q_25, q_50, q_100):
    print('Opening ' + infilename)
    
    #For NetCDF4
    file2read = netCDF4.Dataset(infilename) 
    temp = file2read.variables[id_index]
    ID = temp[:]*1 
    
    temp = file2read.variables[q_max]
    QMax = temp[:]*1 
    
    temp = file2read.variables[q_2]
    Q2 = temp[:]*1 
    
    temp = file2read.variables[q_5]
    Q5 = temp[:]*1 
    
    temp = file2read.variables[q_10]
    Q10 = temp[:]*1 
    
    temp = file2read.variables[q_25]
    Q25 = temp[:]*1 
    
    temp = file2read.variables[q_50]
    Q50 = temp[:]*1 
    
    temp = file2read.variables[q_100]
    Q100 = temp[:]*1 
    
    file2read.close()
    print('Closed ' + infilename)
    
    #This is for NetCDF3
    '''
    file2read = netcdf.NetCDFFile(infilename,'r') 
    
    ID = []
    Q = []
    rivid = file2read.variables[id_index] # var can be 'Theta', 'S', 'V', 'U' etc..
    q = file2read.variables[q_index] # var can be 'Theta', 'S', 'V', 'U' etc..
    n=-1
    for i in rivid:
        n=n+1
        #min_val = min(q[n])
        max_val = max(q[n])
        ID.append(i)
        Q.append(max_val)
    file2read.close()
    '''
    return ID, QMax, Q2, Q5, Q10, Q25, Q50, Q100


def Create_AR_LandRaster(LandCoverFile, LAND_File, projWin_extents, ncols, nrows):
    ds = gdal.Open(LandCoverFile)
    ds = gdal.Translate(LAND_File, ds, projWin = projWin_extents, width=ncols, height = nrows)
    ds = None
    return

def Create_AR_StrmRaster(StrmSHP, STRM_File, outputBounds, minx, miny, maxx, maxy, dx, dy, ncols, nrows, Param):
    source_ds = gdal.OpenEx(StrmSHP)
    gdal.Rasterize(STRM_File, source_ds, format='GTiff', outputType=gdal.GDT_Int32, outputBounds = outputBounds, width = ncols, height = nrows, noData = 0, attribute = Param)
    source_ds = None
    return

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

def Clean_STRM_Raster(STRM_File, STRM_File_Clean):
    print('\nCleaning up the Stream File.')
    (SN, ncols, nrows, cellsize, yll, yur, xll, xur, lat, dem_geotransform, dem_projection) = Read_Raster_GDAL(STRM_File)
    
    #Create an array that is slightly larger than the STRM Raster Array
    B = np.zeros((nrows+2,ncols+2))
    
    #Imbed the STRM Raster within the Larger Zero Array
    B[1:(nrows+1), 1:(ncols+1)] = SN
    (RR,CC) = B.nonzero()
    num_nonzero = len(RR)
    
    for filterpass in range(2):
        #First pass is just to get rid of single cells hanging out not doing anything
        p_count = 0
        p_percent = (num_nonzero+1)/100.0
        n=0
        for x in range(num_nonzero):
            if x>=p_count*p_percent:
                p_count = p_count + 1
                print(' ' + str(p_count), end =" ")
            r=RR[x]
            c=CC[x]
            V = B[r,c]
            if V>0:
                #Left and Right cells are zeros
                if B[r,c+1]==0 and B[r,c-1]==0:
                    #The bottom cells are all zeros as well, but there is a cell directly above that is legit
                    if (B[r+1,c-1]+B[r+1,c]+B[r+1,c+1])==0 and B[r-1,c]>0:
                        B[r,c] = 0
                        n=n+1
                    #The top cells are all zeros as well, but there is a cell directly below that is legit
                    elif (B[r-1,c-1]+B[r-1,c]+B[r-1,c+1])==0 and B[r+1,c]>0:
                        B[r,c] = 0
                        n=n+1
                #top and bottom cells are zeros
                if B[r,c]>0 and B[r+1,c]==0 and B[r-1,c]==0:
                    #All cells on the right are zero, but there is a cell to the left that is legit
                    if (B[r+1,c+1]+B[r,c+1]+B[r-1,c+1])==0 and B[r,c-1]>0:
                        B[r,c] = 0
                        n=n+1
                    elif (B[r+1,c-1]+B[r,c-1]+B[r-1,c-1])==0 and B[r,c+1]>0:
                        B[r,c] = 0
                        n=n+1
        print('\nFirst pass removed ' + str(n) + ' cells')
        
        
        #This pass is to remove all the redundant cells
        n=0
        p_count = 0
        p_percent = (num_nonzero+1)/100.0
        for x in range(num_nonzero):
            if x>=p_count*p_percent:
                p_count = p_count + 1
                print(' ' + str(p_count), end =" ")
            r=RR[x]
            c=CC[x]
            V = B[r,c]
            if V>0:
                if B[r+1,c]==V and (B[r+1,c+1]==V or B[r+1,c-1]==V):
                    if sum(B[r+1,c-1:c+2])==0:
                        B[r+1,c] = 0
                        n=n+1
                elif B[r-1,c]==V and (B[r-1,c+1]==V or B[r-1,c-1]==V):
                    if sum(B[r-1,c-1:c+2])==0:
                        B[r-1,c] = 0
                        n=n+1
                elif B[r,c+1]==V and (B[r+1,c+1]==V or B[r-1,c+1]==V):
                    if sum(B[r-1:r+1,c+2])==0:
                        B[r,c+1] = 0
                        n=n+1
                elif B[r,c-1]==V and (B[r+1,c-1]==V or B[r-1,c-1]==V):
                    if sum(B[r-1:r+1,c-2])==0:
                            B[r,c-1] = 0
                            n=n+1
        print('\nSecond pass removed ' + str(n) + ' redundant cells')
    
    print('Writing Output File ' + STRM_File_Clean)
    Write_Output_Raster(STRM_File_Clean, B[1:nrows+1,1:ncols+1], ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Int32)
    #return B[1:nrows+1,1:ncols+1], ncols, nrows, cellsize, yll, yur, xll, xur
    return

if __name__ == "__main__":
    
    Process_AutoRoute_Geospatial_Data()
    