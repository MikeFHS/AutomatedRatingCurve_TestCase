# local imports
from Automated_Rating_Curve_Generator import calculate_stream_geometry, LinearRegressionPowerFunction

# builtin imports
import csv
import os
import ast

# third party imports
import pandas as pd
import numpy as np
from scipy import stats, interpolate
import matplotlib.pyplot as plt


def parse_array(array_string):
    """Convert a string representation of a list into a NumPy array."""
    # Add commas between numbers
    formatted_string = array_string.replace(' ', '')
    # Use ast.literal_eval to evaluate the string as a list
    return np.array(ast.literal_eval(formatted_string))

def check_for_vdt_outliers(vdt_file, out_vdt_file, xs_file, curve_file, out_curve_file, vdt_steps=15):
    """
    Reads in an ARC VDT Database and Curve File and filters the results for each stream cell based upon:

    1. If the stream cell has a WSE = 0, replace it with a NaN
    2. If the stream cell is a duplicate, based upon duplicate COMID, Row, and Column values, use the one with the lowest WSE.
    3. If the stream cell WSE has a z-score greater than 2 or less than -2, replace it with a NaN. 

    Once filtering is complete, the WSE values for each reach are sorted by elevation. A spline smoothing function
    is then fit to the reaches channel bottom elevation (indepedent variable) and water surface elevation (dependent variable). 
    The spline smoothing leads to a WSE that is smooth from upstream to downstream and in testing, leads to a water surface 
    elevation that either remains constant, or decreases from upstream to downstream. 

    Once the smoothed water surface elevations are calculated for each reach, the script recalculates velocity and top-width based upon
    the stream cells cross-section and the smoothed water surface elevation.

    The script then proceeds to recalculate the rating curves for top-width, elevation, and depth and output a new VDT curve file based upon
    these recalculations.

    Parameters
    ----------
    vdt_file: str
        Path to the ARC generated VDT database file.
    out_vdt_file: str
        Path to the filtered output VDT datafile of this script, the ARC generated VDT database file.
    xs_file: str
        Path to the XS file that is utilized to recalculate V and T with the new interpolated WSE
    curve_file: str
        Path to the ARC generated VDT curve file.
    out_curve_file: str
        Path to the recalculated VDT curve file, one the outputs of this script.
    vdt_steps: int
        The number of individual discharge, velocity, topwidth, and water surface elevation values for each stream cell. The default is 15.
    
    Returns
    -------
    None
    """
    # set this variable equal to true to determine how the output Pandas dataframe is built
    first_vdt_file = True

    # adding column header names for each row of the VDT file
    column_names = ["COMID","Row","Col","Elev","QBaseflow"]
    for step in range(1,vdt_steps+1):     
        Q = f"Q_{step}"
        column_names.append(Q)
        V = f"V_{step}"
        column_names.append(V)
        T = f"T_{step}"
        column_names.append(T)
        WSE = f"WSE_{step}"
        column_names.append(WSE)

    # read the VDT file into Pandas
    vdt_df = pd.read_csv(vdt_file, skiprows=[0], names=column_names)

    # create a list of COMIDs (may not need this)
    comid_list = vdt_df['COMID'].unique().tolist()

    # read the VDT Curve_File
    vdt_curve_df = pd.read_csv(curve_file)

    xs_dict = {}
    # open the cross-section file
    with open(xs_file, mode='r') as file:
        reader = csv.reader(file, delimiter ='\t')
        for row in reader:
            # Parse each column appropriately
            Comid = int(row[0])  # Example for the first column
            Row = int(row[1])  # Example for the second column
            Col = int(row[2])  # Example for the third column
            
            # Extract and parse NumPy arrays
            array1 = parse_array(row[3])
            col5 = float(row[4])  # Example for a float column
            col6 = float(row[5])  # Example for a float column
            array2 = parse_array(row[6])
            array3 = parse_array(row[7])
            col8 = float(row[8])  # Example for a float column
            col9 = float(row[9])  # Example for a float column
            array4 = parse_array(row[10])
            
            # Add the parsed values to the xs_dict for access when we need them
            xs_dict[f"{Comid}_{Row}_{Col}"] = {
                                                'da_xs_profile1': array1,
                                                'd_wse1': col5,
                                                'd_distance_z1': col6,
                                                'dm_manning_n_raster1': array2,
                                                'da_xs_profile2': array3,
                                                'd_wse2': col8,
                                                'd_distance_z2': col9,
                                                'dm_manning_n_raster2': array4
                                               }
            
    # testing to see how many stream cells we flag as bad
    flagged_wse_greater_downstream_than_upstream_length = 0
    flagged_wse_outlier_length = 0
    flagged_wse_zero_length = 0
    total_stream_cell_length = 0
    stream_cells_to_replace_list = []

    # loop through COMIDS
    for comid in comid_list:
        vdt_single_comid_df = vdt_df[vdt_df['COMID'] == comid]
        # loop through each WSE column
        # we're going to use these lists to recaculate the VDT curves
        Q_steps_list = []
        V_steps_list = []
        T_steps_list = []
        WSE_steps_list = []
        for step in range(1,vdt_steps+1):
            WSE_column = f"WSE_{step}"
            WSE_diff_column = f"WSE_{step}_diff"
            WSE_steps_list.append(WSE_column)
        
            # first filter will remove duplicates rows in the VDT and keep the result with the lowest WSE
            # because the issues typically come from moving high-elevation points to low spots.
            vdt_single_comid_df = vdt_single_comid_df.sort_values(by=WSE_column, ascending=True)
            # dropping any duplicate values for each stream cell, just keeping the last one in the list for now
            vdt_single_comid_df = vdt_single_comid_df.drop_duplicates(['COMID','Row','Col'], keep='first')

            # sort the VDT file by descending elevation
            vdt_single_comid_df = vdt_single_comid_df.sort_values(by=['Elev'], ascending=False)

            # assuming that the upstream WSE should be greater than or equal to the downstream WSE for the COMID, 
            # flag if downstream WSE - upstream WSE is > 0
            vdt_single_comid_df[WSE_diff_column] = vdt_single_comid_df[WSE_column] - vdt_single_comid_df[WSE_column].shift(1)

            # testing to see how many stream cells we flag as bad
            flagged_wse_greater_downstream_than_upstream_length = flagged_wse_greater_downstream_than_upstream_length + len(vdt_single_comid_df[(vdt_single_comid_df[WSE_diff_column]>0)])
            total_stream_cell_length = total_stream_cell_length + len(vdt_single_comid_df.index.values)          
            
            # # If the upstream WSE is lower than the downstream, replace it with NaN
            # vdt_single_comid_df.loc[vdt_single_comid_df[WSE_diff_column] > 0, WSE_column] = np.NaN

            # # If the WSE is equal to 0, replace it with NaN
            # flagged_wse_zero_length = flagged_wse_zero_length + ((vdt_single_comid_df[WSE_column]==0)).sum()
            # vdt_single_comid_df.loc[vdt_single_comid_df[WSE_column] == 0, WSE_column] = np.NaN

            # # If the WSE for the stream reach if the z-score is greater than 2 or less-than -2
            # flagged_wse_outlier_length = flagged_wse_outlier_length + (abs(stats.zscore(vdt_single_comid_df[WSE_column])) > 2).sum()
            # vdt_single_comid_df.loc[stats.zscore(abs(vdt_single_comid_df[WSE_column])) > 2, WSE_column] = np.NaN

            # add the section below to test scipy's spline interpolation (3 means cubic spline)
            vdt_single_comid_nan_filtered_df = vdt_single_comid_df.dropna(subset=[WSE_column])
            # dropping any duplicate values for each stream cell, just keeping the last one in the list for now
            vdt_single_comid_nan_filtered_df = vdt_single_comid_nan_filtered_df.drop_duplicates(['Elev'], keep='first')
            # dropping any duplicate values for each stream cell, just keeping the last one in the list for now
            vdt_single_comid_nan_filtered_df = vdt_single_comid_nan_filtered_df.drop_duplicates([WSE_column], keep='first')
            vdt_single_comid_nan_filtered_df = vdt_single_comid_nan_filtered_df.sort_values(by='Elev', ascending=True)

            # fit a spline to the uninterpolated data
            x = vdt_single_comid_nan_filtered_df["Elev"].values
            x = x.astype(float)
            y = vdt_single_comid_nan_filtered_df[WSE_column].values
            y = y.astype(float)
            tck = interpolate.splrep(x, y, k=4, s=len(x)**2)
            # Generate new x values for evaluation
            x_new = np.linspace(min(vdt_single_comid_df['Elev'].values), max(vdt_single_comid_df['Elev'].values), len(vdt_single_comid_df['Elev'].values))
            ynew = interpolate.splev(vdt_single_comid_df['Elev'].values, tck)
            # vdt_single_comid_df[WSE_column] = ynew

            # Use interpolation to fill NaN WSE values
            # vdt_single_comid_df[WSE_column] = vdt_single_comid_df[WSE_column].interpolate(method='linear')
            
            # remove NaNs where interpolation doesn't work
            vdt_single_comid_df = vdt_single_comid_df.dropna(subset=[WSE_column])

            # drop the difference column, we don't need this anymore
            vdt_single_comid_df = vdt_single_comid_df.drop(columns=[WSE_diff_column])

            top_widths = []
            velocities = []
            # now we'll loop through and replace the V and TW for the streams we've interpolated for
            for index, row in vdt_single_comid_df.iterrows():
                stream_cell_comid = int(row['COMID'])
                stream_cell_row = int(row['Row'])
                stream_cell_col = int(row['Col'])
                stream_cell_xs_dict = xs_dict[f"{stream_cell_comid}_{stream_cell_row}_{stream_cell_col}"]
                wse_new = row[f"{WSE_column}"]
                A1, P1, R1, np1, T1 = calculate_stream_geometry(stream_cell_xs_dict['da_xs_profile1'],wse_new,stream_cell_xs_dict['d_distance_z1'],stream_cell_xs_dict['dm_manning_n_raster1'])
                A2, P2, R2, np2, T2 = calculate_stream_geometry(stream_cell_xs_dict['da_xs_profile2'],wse_new,stream_cell_xs_dict['d_distance_z2'],stream_cell_xs_dict['dm_manning_n_raster2'])
                # Aggregate the geometric properties
                top_width = float(T1 + T2)
                area = float(A1 + A2)
                if area <= 0 or top_width <= 0:
                    velocity = np.nan
                    top_width = np.nan
                else:
                    Q_column = f"Q_{step}"
                    velocity = round(float(row[f'{Q_column}'])/float(area),3)
                top_widths.append(top_width)
                velocities.append(velocity)

            vdt_single_comid_df[f"T_{step}"] = top_widths
            vdt_single_comid_df[f"V_{step}"] = velocities

            Q_steps_list.append(Q_column)
            T_steps_list.append(f"T_{step}")
            V_steps_list.append(f"V_{step}")

            # We've started building the new VDT file
            if first_vdt_file is True:
                first_vdt_file = False
                output_vdf_df = vdt_single_comid_df
            else:
                output_vdf_df = pd.concat([output_vdf_df,vdt_single_comid_df])
    
            # # Plot the fitted spline
            # distance = np.arange(0, 10*len(vdt_single_comid_df['Elev']), 10)
            # plt.plot(distance, vdt_single_comid_df['Elev'], label='Channel Bottom Elevation', color='brown')

            # print(vdt_single_comid_df[WSE_column])
            # # Plot the original data
            # plt.plot(distance, vdt_single_comid_df[WSE_column], label='Source VDT Data without filtering', color='red')
            # # Plot the original data
            # plt.plot(distance, ynew, label='Spline Fitted WSE without linear filling', color='green')

            # # Add labels and legend
            # plt.xlabel('Distance')
            # plt.ylabel('Elevation (m)')
            # plt.legend()

            # # Show the plot
            # plt.show()
            
    # now let's recaculate the VDT curves for the stream
    # these list will be used to build the output dataframe for the curve file
    COMID_list = []
    Row_list = []
    Col_list = []
    BaseElev_list = []
    DEM_Elev_list = []
    QMax_list = []
    depth_a_list = []
    depth_b_list = []
    tw_a_list = []
    tw_b_list = []
    vel_a_list = []
    vel_b_list = []
    for index, row in output_vdf_df.iterrows():
        # Create list for the current row 
        Q_list = []
        WSE_list = []
        T_list = []
        V_list = []
        for Q_step in Q_steps_list:
            Q_list.append(row[Q_step])
        for WSE_step in WSE_steps_list:
            WSE_list.append(row[WSE_step])
        for T_step in T_steps_list:
            T_list.append(row[T_step])    
        for V_step in V_steps_list:
            V_list.append(row[V_step]) 

        # recalculate the velocity curve
        (d_v_a, d_v_b, d_v_R2) = LinearRegressionPowerFunction(np.array(Q_list), np.array(V_list))
        if d_v_a == -9999.9 or d_v_b == -9999.9:
            pass
        else:
            # recalculate the top-width curve
            (d_t_a, d_t_b, d_t_R2) = LinearRegressionPowerFunction(np.array(Q_list), np.array(T_list))
            D_list = [WSE - row['Elev'] for WSE in WSE_list]
            # recalculate the depth curve
            (d_d_a, d_d_b, d_d_R2) = LinearRegressionPowerFunction(np.array(Q_list), np.array(D_list))
            try:
                vdt_curve_filtered_df = vdt_curve_df.loc[(vdt_curve_df['COMID'] == row['COMID']) & (vdt_curve_df['Row'] == row['Row']) & (vdt_curve_df['Col'] == row['Col'])]
                print(vdt_curve_filtered_df)
                DEM_Elev = vdt_curve_filtered_df['DEM_Elev'].values[0]
                Qmax = vdt_curve_filtered_df['QMax'].values[0]
                # append all the outputs to the lists
                COMID_list.append(row['COMID'])
                Row_list.append(row['Row'])
                Col_list.append(row['Col'])
                BaseElev_list.append(row['Elev'])
                DEM_Elev_list.append(DEM_Elev)
                QMax_list.append(Qmax)
                depth_a_list.append(round(d_d_a, 3))
                depth_b_list.append(round(d_d_b, 3))
                tw_a_list.append(round(d_t_a, 3))
                tw_b_list.append(round(d_t_b, 3))
                vel_a_list.append(round(d_v_a, 3))
                vel_b_list.append(round(d_v_b, 3))
            except:
                print(f"Didn't work for {row['COMID']} {row['Row']} {row['Col']}")

    
    # built the new output curve file
    vdt_curve_df = pd.DataFrame({'COMID': COMID_list,
                                'Row': Row_list,
                                'Col': Col_list,
                                'BaseElev': BaseElev_list,
                                'DEM_Elev': DEM_Elev_list,
                                'QMax': QMax_list,
                                'depth_a': depth_a_list,
                                'depth_b': depth_b_list,
                                'tw_a': tw_a_list,
                                'tw_b': tw_b_list,
                                'vel_a': vel_a_list,
                                'vel_b': vel_b_list
                                })
    # output the curve to csv
    vdt_curve_df.to_csv(out_curve_file, index = False)
            


    # ensure row and column remain integers
    output_vdf_df = output_vdf_df.astype({"Row": 'Int64', "Col": 'Int64'})
    
    # delete the old filtered VDT file, if it already exists
    if os.path.exists(out_vdt_file):
        os.remove(out_vdt_file)
    else:
        pass

    # now we add the headers back to the VDT file here
    header_list = ['COMID','Row','Col','Elev','QBaseflow','Q','V','T','WSE']
    with open(out_vdt_file, mode ='w') as csvfile:  
      
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)  
    
        # writing the fields
        csvwriter.writerow(header_list)

        # output list of lists to input into csv file
        values = output_vdf_df.values.tolist()

        # writing the fields
        csvwriter.writerows(values)
    
    # here's what percentage of cells we filtered based upon WSE not decreasing with elevation
    percent_filtered_for_wse_higher_downstream_than_upstream =  round(flagged_wse_greater_downstream_than_upstream_length / total_stream_cell_length * 100,2)
    print(f"{percent_filtered_for_wse_higher_downstream_than_upstream}% were filtered because downstream WSE was greater than upstream WSE (measure by decreasing Elevation for each stream)...\n")
    # here's what percentage of cells we filtered based upon WSE being an outlier for the stream
    percent_filtered_for_wse_outliers = round(flagged_wse_outlier_length/ total_stream_cell_length * 100,2)
    print(f"{percent_filtered_for_wse_outliers}% were filtered because WSE at stream cell was an outlier (z score < 2 or z score > 2)...\n")
    # here's what percentage of cells we filtered because we found a stream cell with WSE = 0
    percent_filtered_for_wse_zero_length = round(flagged_wse_zero_length/ total_stream_cell_length * 100,2)
    print(f"{percent_filtered_for_wse_zero_length}% were filtered because WSE for a stream cell equaled zero...\n")
    return 


if __name__ == "__main__":

    vdt_file = r"C:\Users\jlgut\OneDrive\Desktop\AutomatedRatingCurve_TestCase\VDT\Gardiner_VDT_Database.txt"
    out_vdf_file = r"C:\Users\jlgut\OneDrive\Desktop\AutomatedRatingCurve_TestCase\VDT\Gardiner_VDT_Database_filtered.txt"
    xs_file = r"C:\Users\jlgut\OneDrive\Desktop\AutomatedRatingCurve_TestCase\XS\Gardiner_XS_File.txt"
    curve_file = r"C:\Users\jlgut\OneDrive\Desktop\AutomatedRatingCurve_TestCase\VDT\Gardiner_CurveFile.csv"
    out_curve_file = r"C:\Users\jlgut\OneDrive\Desktop\AutomatedRatingCurve_TestCase\VDT\Gardiner_CurveFile_filtered.csv"
    vdt_steps = 15
    check_for_vdt_outliers(vdt_file, out_vdf_file, xs_file, curve_file, out_curve_file, vdt_steps)