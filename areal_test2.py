import os
from SampleDataLoader import load_flood_data
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

def water_shed_finder(data_frame, watershed_file=None, remove_singular=True):
    '''
    for each location, find which watershed it belongs to
    Args:
        data_frame (pandas dataframe) : Must have a LONGTITUDE column
            and a LATITUDE column
        watershed_file (string):  usually a file with extension shp, if not
            specified, the default watershed file: HUC8 in South Caorolina, will be
            used.
        remove_singular (boolean): if true, the function will get rid of
        locations without neighbors.
    return
        A new dataframe with an additional column called WATERSHED
    '''
    if watershed_file is None:
        print('')
        watershed_file = 'data/shape_file/hydrologic_HUC8_units/wbdhu8_a_sc.shp'
        watershed_file = os.path.join(os.getcwd(), watershed_file)

    huc8_units = gpd.read_file(watershed_file)
    water_shed_info = huc8_units[['NAME', 'geometry']]

    names = []
    df = data_frame.copy()

    locs = df[['LONGITUDE', 'LATITUDE']]
    for _, loc in locs.iterrows():
        for _, rows in water_shed_info.iterrows():
            name, polygon = rows
            if Point(loc).within(polygon):
                names.append(name)
                break
        else:
            names.append('HUC Not Defined')
    df['WATERSHED'] = names

    if remove_singular:
        number_of_obs = df.groupby(['WATERSHED']).count()['SITENUMBER']
        singular_huc_areas = number_of_obs[number_of_obs==1].index
        df = df[~df.WATERSHED.isin(singular_huc_areas)]
    return df

def neighbor_finder(dataframe, column):

    watershed_list = dataframe[column]
    neighbor_matrix = [] # an nxn matrix, 1 for adjcent, 0 for not.
    for entry in watershed_list:
        w = [0 if entry!=comp else 1 for comp in watershed_list]
        neighbor_matrix.append(w)
    neighbor_matrix = np.array(neighbor_matrix)

    adjcent_matrix = [] # a list of neighbors for each loc
    weight_matrix = []
    for idx, row in enumerate(neighbor_matrix):
        mask = np.argwhere(row == 1).ravel().tolist()
        mask.remove(idx) #delete the location itself.
        adjcent_matrix.append(mask)
        weight_matrix.append([1]*len(mask))

    return neighbor_matrix, adjcent_matrix, weight_matrix

if __name__ == '__main__':
    daily_flood = load_flood_data(option = 'daily')#usage
    df_with_watershed_assigned = water_shed_finder(daily_flood)
    NM, AM, WM = neighbor_finder(df_with_watershed_assigned, 'WATERSHED')
    print(WM, AM, NM)
