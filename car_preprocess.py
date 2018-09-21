import os
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

class ArealDataPreprocessor:

    def __init__(self, spatial_data_frame, watershed_file=None, remove_singular=True, key='STATION'):

        self._spatial_data_frame = spatial_data_frame
        self.remove_singular = remove_singular
        self.key = key

        if watershed_file is None:
            path = 'data/shape_file/hydrologic_HUC8_units/wbdhu8_a_sc.shp'
            self.watershed_file = os.path.join(os.getcwd(), path)
        else:
            self.watershed_file = watershed_file

        self._method_dispatcher()

    def _lookup_watershed(self):
        '''
        for each location, find which watershed it belongs to.
        A attribute 'data' will be generated which is a new
        dataframe with an additional column called WATERSHED
        '''
        huc8_units = gpd.read_file(self.watershed_file)
        water_shed_info = huc8_units[['NAME', 'geometry']]

        names = []
        dataframe_sp = self._spatial_data_frame.copy()
        locs = dataframe_sp[['LONGITUDE', 'LATITUDE']]

        for _, loc in locs.iterrows():
            for _, rows in water_shed_info.iterrows():
                name, polygon = rows
                if Point(loc).within(polygon):
                    names.append(name)
                    break
            else:
                names.append('undefined watershed')

        dataframe_sp['WATERSHED'] = names

        if self.remove_singular:

            number_of_obs = dataframe_sp.groupby(['WATERSHED']).count()[self.key]
            singular_huc_areas = number_of_obs[number_of_obs==1].index
            dataframe_sp = dataframe_sp[~dataframe_sp.WATERSHED.isin(singular_huc_areas)]
        print('created a new column called WATERSHED to store the huc watershed information')
        return dataframe_sp

    def _generate_adjacent_matrix(self):

        '''
        find the adjacent matrix and weight matrix for each location
        based on the dummy variable of region ('WATERSHED')
        returns:
            adjacent_matrix, index for neighbors. [2, 4] means the touch border with 3rd and 5th observation
            weight_matrix: replace all elements in adjacent matrix with 1.
        '''
        watershed_list = self.data['WATERSHED']
        neighbor_matrix = [] # an n by n matrix, 1 for adjcent, 0 for not.
        adjacent_matrix = []
        weight_matrix = []

        for entry in watershed_list:
            w = [0 if entry!=comp else 1 for comp in watershed_list]
            neighbor_matrix.append(w)
        neighbor_matrix = np.array(neighbor_matrix)

        for idx, row in enumerate(neighbor_matrix):
            mask = np.argwhere(row == 1).ravel().tolist()
            mask.remove(idx) #delete the location itself.

            adjacent_matrix.append(mask)
            weight_matrix.append([1]*len(mask))
        return adjacent_matrix, weight_matrix

    def _generate_padded_matrix(self, input_list):
        '''
        construct the matrix of W by making sure that every
        row has the same length.
        args:
            input list (list): a list of spatial information for each location
        return:
            the padded matrix with rows of equal length.
        '''
        max_vector_length = max([len(row) for row in input_list])
        input_list_copy = input_list.copy()

        for row in input_list_copy:
            while len(row) < max_vector_length:
                row.append(0)
        return np.array(input_list_copy)

    def _method_dispatcher(self):
        '''
        make available 4 attributes for the object
            data: dataframe with one more column of watershed
            adjacent_matrix: index of neighbors
            weight_matrix: weights of neighbors
            padded_weight_matrix: weights of neighbors, length of each row is the same
            padded_adjacent_matrix: index of neighbors, padded with 0 to make sure equal length for each row

        '''
        self.data = self._lookup_watershed()
        self.adjacent_matrix, self.weight_matrix = self._generate_adjacent_matrix()
        self.padded_weight_matrix = self._generate_padded_matrix(self.weight_matrix)
        self.padded_adjacent_matrix = self._generate_padded_matrix(self.adjacent_matrix)

if __name__ == '__main__':

    from SampleDataLoader import load_flood_data
    daily_flood = load_flood_data(option='daily')
    areal_data = ArealDataPreprocessor( daily_flood, key='SITENUMBER')
    print(areal_data.data)
    print(areal_data.adjacent_matrix)
    print(areal_data.weight_matrix)
    print(areal_data.padded_weight_matrix)
    print(areal_data.padded_adjacent_matrix)



# def water_shed_lookup(data_frame, watershed_file=None, remove_singular=True, key='STATION'):
#     '''
#     for each location, find which watershed it belongs to
#     Args:
#         data_frame (pandas dataframe) : Must have a LONGTITUDE column
#             and a LATITUDE column
#         watershed_file (string):  usually a file with extension shp, if not
#             specified, the default watershed file: HUC8 in South Caorolina, will be
#             used.
#         remove_singular (boolean): if true, the function will get rid of
#         locations without neighbors.
#     return
#         A new dataframe with an additional column called WATERSHED
#     '''
#     if watershed_file is None:
#         watershed_file = 'data/shape_file/hydrologic_HUC8_units/wbdhu8_a_sc.shp'
#         watershed_file = os.path.join(os.getcwd(), watershed_file)

#     huc8_units = gpd.read_file(watershed_file)
#     water_shed_info = huc8_units[['NAME', 'geometry']]

#     names = []
#     df = data_frame.copy()

#     locs = df[['LONGITUDE', 'LATITUDE']]
#     for _, loc in locs.iterrows():
#         for _, rows in water_shed_info.iterrows():
#             name, polygon = rows
#             if Point(loc).within(polygon):
#                 names.append(name)
#                 break
#         else:
#             names.append('HUC Not Defined')
#     df['WATERSHED'] = names

#     if remove_singular:
#         number_of_obs = df.groupby(['WATERSHED']).count()[key]
#         singular_huc_areas = number_of_obs[number_of_obs==1].index
#         df = df[~df.WATERSHED.isin(singular_huc_areas)]
#     return df

# def generate_adjacent_matrix(dataframe, column):
#     '''
#     find the adjacent matrix and weight matrix for each location
#     based on the dummy variable of region
#     args:
#         dataframe (pandas dataframe): a dataframe with a column of regional information
#         column (string): the name of dummy variable that contains regional info
#     return:
#         three lists:
#             neighbor_matrix: an n by n matrix with binaries, 0 means not adjacent, 1 otherwise
#             adjacent_matrix, index for neighbors. [2, 4] means the touch border with 3rd and 5th observation
#             weight_matrix: replace all elements adjacent matrix with 1.
#     '''
#     watershed_list = dataframe[column]
#     neighbor_matrix = [] # an n by n matrix, 1 for adjcent, 0 for not.
#     for entry in watershed_list:
#         w = [0 if entry!=comp else 1 for comp in watershed_list]
#         neighbor_matrix.append(w)
#     neighbor_matrix = np.array(neighbor_matrix)

#     adjacent_matrix = [] # a list of neighbors for each loc
#     weight_matrix = []
#     for idx, row in enumerate(neighbor_matrix):
#         mask = np.argwhere(row == 1).ravel().tolist()
#         mask.remove(idx) #delete the location itself.
#         adjacent_matrix.append(mask)
#         weight_matrix.append([1]*len(mask))

#     return neighbor_matrix, adjacent_matrix, weight_matrix


# def matrix_padding(input_list, max_padding=None):
#     '''
#     construct the matrix of W by making sure that every
#     row has the same length.
#     args:
#         input list (list): a list of spatial information for each location
#         max padding (int): the length of padded row. Must be equal or longer than the longest row.
#     '''
#     max_vector_length = max([len(row) for row in input_list])
#     if max_padding is None:
#         max_padding = max_vector_length
#     elif max_padding > max_vector_length:
#         pass
#     else:
#         raise ValueError('max_padding must be equal or greater than the longest row')

#     input_list_copy = input_list.copy()
#     for row in input_list_copy:
#         while len(row) < max_padding:
#             row.append(0)

#     return np.array(input_list_copy)

# def merge_rain_and_flood(flood_data, rainfall_data, verbose=False):

#     '''
#     a brute force method to match locations
#     based on np.close(a,b, precision)
#     a and b will be considered as close enough if the diff is within the specified precison

#     '''
#     flood_data_copy = flood_data.copy()
#     nearest_rainfall_info = []

#     for index, row in flood_data.iterrows():

#         if verbose:
#             print(f'starting analyzing row {index}')

#         lat_comp = row['LATITUDE']
#         lon_comp = row['LONGITUDE']

#         valid_rain_value = None
#         precision = 0.01

#         while not valid_rain_value:

#             if verbose:
#                 print(f'testing the precision {precision}')

#             for idx, row in rainfall_data.iterrows():
#                 lat, lon, rain = row
#                 lat_equal = np.isclose(lat,lat_comp, atol=precision)
#                 lon_equal = np.isclose(lon,lon_comp, atol=precision)

#                 if lat_equal and lon_equal:

#                     valid_rain_value = rain
#                     nearest_rainfall_info.append(valid_rain_value)
#                     break

#             precision = precision + 0.05

#     flood_data_copy['PRCP'] = nearest_rainfall_info
#     return flood_data_copy


# if __name__ == '__main__':
#     daily_flood = load_flood_data(option='daily')
#     flood_with_watershed = water_shed_lookup(daily_flood, key='SITENUMBER')

    # f_NM, f_AM, f_WM = generate_adjacent_matrix(flood_with_watershed, 'WATERSHED')
    # padded_weights_f = matrix_padding(f_AM)
    # padded_adj_f = matrix_padding(f_WM)

    # daily_rain = load_rainfall_data(option='daily')
    # rain_with_watershed = water_shed_lookup(daily_rain, key='STATION')
    # r_NM, r_AM, r_WM = generate_adjacent_matrix(rain_with_watershed, 'WATERSHED')
    # padded_weights_r = matrix_padding(r_AM)
    # padded_adj_r = matrix_padding(r_WM)

