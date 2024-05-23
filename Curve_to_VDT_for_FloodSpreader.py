#Using the curve data from ARC, this simple script creates a VDT input file for FloodSpreader (https://erdc-library.erdc.dren.mil/jspui/handle/11681/38783)
#Written on 5/22/2024 by Mike Follum, Follum Hydrologic Solutions, LLC.

#conda env create -f environment.yml

import sys
import os
import numpy as np


def power_func(x, a, b):
    return a * np.power(x, b)

def Read_Flow_File(Input_Flow_File):
    infile = open(Input_Flow_File,'r')
    lines = infile.readlines()
    infile.close()
    num_lines = len(lines)
    
    #Find min and max COMID
    min_COMID = 99999999999999
    max_COMID = -99999
    for i in range(1,num_lines):
        ls = lines[i].strip().split(',')
        C = int(ls[0])
        if C<min_COMID:
            min_COMID = C
        if C>max_COMID:
            max_COMID = C
    FF_COMID = np.zeros(max_COMID-min_COMID + 1)
    FF_Q = np.zeros(max_COMID-min_COMID + 1)
    for i in range(1,num_lines):
        ls = lines[i].strip().split(',')
        C = int(ls[0])
        FF_COMID[C-min_COMID] = int(ls[0])
        FF_Q[C-min_COMID] = float(ls[1])
    return FF_COMID, FF_Q, min_COMID, max_COMID

if __name__ == "__main__":
    Input_Curve_File = 'VDT/Gardiner_CurveFile.csv'
    Input_Flow_File = 'FlowFile/COMID_Q_qout_max.txt'
    Output_VDT_File = 'VDT/Gardiner_VDT_FS.csv'
    
    (FF_COMID, FF_Q, min_COMID, max_COMID) = Read_Flow_File(Input_Flow_File)
    
    #Open the Curve File
    infile = open(Input_Curve_File,'r')
    lines = infile.readlines()
    infile.close()
    num_lines = len(lines)
    
    #Open the Output VDT File
    print('Working on writing data to ' + Output_VDT_File)
    outfile = open(Output_VDT_File, 'w')
    outfile.write('COMID,Row,Col,Q,V,D,T,Elev,WSE')
    
    #Now go through each line and create a simple VDT File for FloodSpreader
    for i in range(1,num_lines):
        (COMID,Row,Col,BaseElev,DEM_Elev,QMax,depth_a,depth_b,tw_a,tw_b,vel_a,vel_b) = lines[i].strip().split(',')
        Q = FF_Q[int(COMID)-min_COMID]
        V = power_func(Q, float(vel_a), float(vel_b))
        D = power_func(Q, float(depth_a), float(depth_b))
        TW = power_func(Q, float(tw_a), float(tw_b))
        WSE = float(BaseElev) + D
        DEM_Elev = float(DEM_Elev)
        if DEM_Elev>0.0 and WSE<=DEM_Elev:
            D = 0.01
            WSE = DEM_Elev + D
        out_str = '\n' + COMID + ',' + Row + ',' + Col + ",{:.3f}".format(Q) + ",{:.3f}".format(V) + ",{:.3f}".format(D) + ",{:.3f}".format(TW) + ',' + BaseElev + ",{:.3f}".format(WSE)
        outfile.write(out_str)
    outfile.close()
    print('Finished')
        