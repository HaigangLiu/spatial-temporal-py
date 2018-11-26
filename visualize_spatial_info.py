import os
import webbrowser
import shapely
from folium import Map, Marker, CircleMarker
from folium.plugins import MarkerCluster
from folium.features import PolygonMarker
from utility_functions import get_state_contours, get_state_fullname

class SpatialPlotter:
    '''
    A class helps visualize commonly used spatial elements: locations, countours and points.
    Args:
        central_point (list): the central point (lat and lon coordinate) for the canvass.
            Note: users can pass a list of points and we will use the mean value as central point
        reverse (boolean): if true, the position of lat and lon will be reversed.
    '''
    def __init__(self, central_point, reverse=False):
        if reverse:
            central_point = self._reverse_lat_lon(central_point)
        self._build_canvass(central_point)
        print('-'*20)
        print('Latitude is assumed to be the first column')
        print('if your data has longitude first. set reverse=True')

    def _build_canvass(self, locations):
        lat_sum = 0; lon_sum = 0
        for row in locations:
            lat_sum = lat_sum + row[0]
            lon_sum = lon_sum + row[1]
        average_lat = lat_sum/len(locations)
        average_lon = lon_sum/len(locations)
        center = [average_lat, average_lon]
        self.canvass = Map(location=center, zoom_start=6)

    def _reverse_lat_lon(self, list_of_coords):
        '''
        allow users to flip the order of latitude and longitude in the list
        Conventionally, use latitude as the first argument unless specified otherwise
        '''
        flipped_locations = []
        for coord in list_of_coords:
            new_loc = list(reversed(coord[0:2])) #only reverse lat and lon
            new_loc.extend(coord[2:])
            flipped_locations.append(new_loc)
        return flipped_locations

    def _pandas_to_list(self, locations):
        '''
        if input is dataframe, this function will convert the input to list
        '''
        try:
            locations = locations.values.tolist()
        except:
            if isinstance(locations, list):
                pass
            else:
                raise TypeError('Acceptable input types: list and dataframe')
        return locations

    def add_point(self, points):
        '''
        Add location markers on the canvass
        Args:
            points: the points to be added. We assume the first two dimension are locational information.
            The orders has to be latitude, longitude
        '''
        points = self._pandas_to_list(points)
        for point in points:
            Marker(location=point[0:2]).add_to(self.canvass)
        return self

    def add_point_clustered(self, points):
        '''
        Add location markers, but will automatically make cluster if there is too many.
        Args:
            points: the points to be added. We assume the first two dimension are locational information
        '''
        points = self._pandas_to_list(points)
        marker_cluster = MarkerCluster(icons="dd").add_to(self.canvass)
        for point in points:
            Marker(location = point[0:2]).add_to(marker_cluster)
        return self

    def add_contour(self, contour='SC'):
        '''
        Add the contour information on the canvass
        Args:
            contour: allow three types of information:
                1.Statenames: like SC, NC or north carolina
                2.shapely polygon type
                3.list of coords
        '''
        print('-'*20)
        print('allow two input types: 1. eg. DC, SC 2. a list of coordinates')

        if isinstance(contour, str):
            polygon = get_state_contours(contour)[-1]
            make_coords = True

        elif isinstance(contour, shapely.geometry.polygon.Polygon):
            polygon = contour
            make_coords = True

        elif isinstance(contour, list):
            make_coords = False

        else:
            raise TypeError('only support str, list and polygon type')

        if make_coords:
            longitudes, latitudes = polygon.exterior.coords.xy
            list_of_coords = list(zip(latitudes, longitudes))

        PolygonMarker(list_of_coords, color='blue', fill_opacity=0.2, weight=1).add_to(self.canvass)

        return self

    def add_value(self, values, multiplier=4):
        values = self._pandas_to_list(values)
        for record in values:
            value = record[2]; location_info = record[0:2]
            color = "#ff0000" if value >=0 else "#000000" # pos and neg have different colors
            CircleMarker(location=location_info,
                         radius=multiplier*abs(value),
                         alpha=0.5,
                         fill=True,
                         fill_color=color,
                         color=color).add_to(self.canvass)
        return self

    def plot(self, open_in_browser=True, filename=None):
        if filename is None:
            filename = 'map_test.html'

        output_path = os.path.join(os.getcwd(), filename)
        self.canvass.save(output_path)
        print('-'*20)
        print(f'the map has been saved to {filename}')

        if filename == 'map_test.html':
            print('to change to a different name, assign a name to filename')
            print('when calling plot() function')
        if open_in_browser:
            webbrowser.open('file://' + os.path.realpath(output_path))

if __name__ == '__main__':
    #the test based on list input
    s = SpatialPlotter([[34, -80]])
    s.add_point([[33.9, -80], [33.8, -80], [33.2, -80]])\
     .add_contour('North Carolina')\
     .add_value([[33.9, -80, 1], [34.9, -80, 20]])\
     .plot()

    # #the test case based on dataframe input
    # from SampleDataLoader import load_rainfall_data
    # test = load_rainfall_data('monthly')
    # new_map = SpatialPlotter([[34, -80]])\
    #             .add_point(test[['LATITUDE', 'LONGITUDE']])\
    #             .plot(filename='map_test2.html')
