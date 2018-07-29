import fiona
import folium
import numpy as np
import pandas as pd
import os
import webbrowser
import pickle
from spatial_model_gp import GPModelSpatial
import matplotlib.pyplot as plt

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

    def __init__(self, locations, variable = None, map_file = None):

        self.locations = locations[['LATITUDE', 'LONGITUDE']]
        self.variable = variable

        if map_file is None:
            self.map_file = os.path.join(os.getcwd(),'data/shape_file/south_carolina/tl_2010_45_state10.shp')
        else:
            self.map_file = map_file
        shape = fiona.open(self.map_file)

        self.shape_info = shape.next()
        self.center= list(self.locations.median(axis = 0))

        self.locs = list(self.locations.values)
        self.map = folium.Map(location= self.center, zoom_start = 7)

    def _save_map(self, figname, open_in_browser):
        if figname:
            assert figname.endswith('html'), 'need .html as file extension'
            self.map.save(figname)

        else:
            figname = 'test_map.html'
            self.map.save(figname)

        output_path = os.path.join(os.getcwd(), figname)
        print(f'the map has been save to {output_path}')
        if open_in_browser:
            webbrowser.open('file://' + os.path.realpath(output_path))


    def plot_locations(self, unclustered = True, contoured = True, open_in_browser = True, figname = None):

        if unclustered:
            for coord in self.locs:
                folium.Marker(location= coord).add_to(self.map)
        else:
            marker_cluster = MarkerCluster(icons="dd").add_to(self.map)
            for coord in self.locs:
                folium.Marker(location = coord).add_to(marker_cluster)

        if contoured:
            folium.GeoJson(
                self.shape_info,
                style_function = lambda feature: {
                    'weight' : 1,
                    'fillOpacity' : 0.1,
                    }
                ).add_to(self.map)

        self._save_map(figname, open_in_browser)


        return self.map

    def plot_variable_values(self, contoured = True, open_in_browser = True, size_multiplier = 1, figname = None):

        for value, location in zip(self.variable, self.locations.values):

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

        self._save_map(figname, open_in_browser)
        return self.map

def diagnosis_plot_error_bars_spatial(spatial_model, upper_percentile = 95, lower_percentile = 5, figname = None):

    simulated_values = spatial_model.simulated_values['y']
    real_values = spatial_model.Y_new
    predictions = spatial_model.predictions

    upper_bound = np.exp(np.percentile(simulated_values, upper_percentile, axis = 0))
    lower_bound = np.exp(np.percentile(simulated_values, lower_percentile, axis = 0))

    x = list(range(len(real_values)))
    errorbar_length = 0.5*(upper_bound - lower_bound)

    fig = plt.figure()
    plt.plot(x, real_values, "ro", alpha = 0.5)
    plt.errorbar(x, predictions, errorbar_length, fmt = "o", alpha = 0.5)

    if figname:
        assert figname.endswith('.png'), 'needs a .png file extension.'
        fig.savefig(figname)
        print(f'the traceplot has been saved to {os.path.join(os.getcwd(), figname)}')
    return fig


if __name__ == '__main__':

    # ------ this is for raw data -------
    # from SampleDataLoader import load_rainfall_data
    # r = load_rainfall_data()
    # plotter = PlotDataOnMap(r[['LATITUDE', 'LONGITUDE']],r.PRCP )
    # s1 = plotter.plot_locations()
    # s2 = plotter.plot_variable_values(size_multiplier = 2)

    # ------ this is for residuals data -------

    from SampleDataLoader import load_rainfall_data
    with open('result.pickle', 'rb') as handler:
        result_from_pickle = pickle.load(handler)

    loc = result_from_pickle.test_loc_cache
    predictions = result_from_pickle.predictions
    real_values = result_from_pickle.Y_new

    locs_ = PlotDataOnMap(variable = predictions - real_values, locations = loc).plot_locations(figname = 'just_location.html')

    values_ = PlotDataOnMap(variable = predictions - real_values, locations = loc).plot_variable_values(size_multiplier = 3.5, figname = 'with_value.html')

    fig2 = diagnosis_plot_error_bars_spatial(result_from_pickle, figname = 'test.png')

    plt.show()
    print(predictions)
    print(real_values)
