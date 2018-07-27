import fiona
import folium
import numpy as np
import pandas as pd
import os
import webbrowser

class PlotDataOnMap:
    '''
    Visualising locations on a map, or plot the spatially-vaying
    data on the map. The size of point is proportional to the value.

    Args:
        locations(dataframe): a dataframe with LATITUDE and LONGITTUDE column
        map_file(shapefile): shapefile users can download from USGS
                            default one is South Carolina
        variable(pandas series): the spatially indexed variable. e.g. rainfall value or flood.
    '''

    def __init__(self, locations, map_file = None):

        self.locations = locations[['LATITUDE', 'LONGITUDE']]
        if map_file is None:
            self.map_file = '/Users/haigangliu/SpatialTemporalBayes/data/shape_file/south_carolina/tl_2010_45_state10.shp'
        else:
            self.map_file = map_file
        shape = fiona.open(self.map_file)

        self.shape_info = shape.next()
        self.center= list(self.locations.median(axis = 0))

        self.locs = list(self.locations.values)
        self.map = folium.Map(location= self.center, zoom_start = 7)

    def plot_locations(self, unclustered = True, contoured = True, open_in_browser = True):

        if unclustered:
            for coord in self.locs:
                folium.Marker(location= coord).add_to(self.map)
        else:
            marker_cluster = MarkerCluster(icons="dd").add_to(self.map)
            for coord in self.locs:
                folium.Marker(location=[ coord[0], coord[1]]).add_to(marker_cluster)

        if contoured:
            folium.GeoJson(
                self.shape_info,
                style_function = lambda feature: {
                    'weight' : 1,
                    'fillOpacity' : 0.1,
                    }
                ).add_to(self.map)

        self.map.save('test_map.html')
        output_path = os.path.join(os.getcwd(), 'test_map.html')
        print(f'the map has been save to {output_path}')
        if open_in_browser:
            webbrowser.open('file://' + os.path.realpath(output_path))
        return self.map

    def plot_variable_values(self, variable, contoured = True, open_in_browser = True, size_multiplier = 1):

        for value, location in zip(variable, self.locations.values):

            if value >= 0:
                folium.CircleMarker(location= location,
                                    radius = size_multiplier*np.abs(value),
                                    alpha = 0.5,
                                    fill= True,
                                    fill_color = "#ff0000",
                                    color="#ff0000").add_to(self.map)
            else: # positive and negative come in different colors
                folium.CircleMarker(location=location,
                                    radius = size_multiplier*np.abs(value),
                                    alpha = 0.5,
                                    fill= True,
                                    fill_color = "#000000",
                                    color="#000000").add_to(self.map)

        if contoured:
            folium.GeoJson(
                self.shape_info,
                style_function = lambda feature: {
                    'weight' : 1,
                    'fillOpacity' : 0.1,
                    }
                ).add_to(self.map)

        self.map.save('test_map_with_value.html')
        output_path = os.path.join(os.getcwd(), 'test_map_with_value.html')

        print(f'the map has been save to {output_path}')
        if open_in_browser:
            webbrowser.open('file://' + os.path.realpath(output_path))
        return self.map

if __name__ == '__main__':
    from SampleDataLoader import load_rainfall_data
    r = load_rainfall_data()
    plotter = PlotDataOnMap(r[['LATITUDE', 'LONGITUDE']])
    s1 = plotter.plot_locations()
    s2 = plotter.plot_variable_values(variable = r['PRCP'], size_multiplier = 2)
