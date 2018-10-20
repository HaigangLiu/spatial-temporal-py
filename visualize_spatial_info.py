import os
import webbrowser
from folium import Map, Marker, CircleMarker
from folium.plugins import MarkerCluster
from folium.features import PolygonMarker
from utility_functions import get_state_contours

class SpatialPlotter:
    #ask user to provide a central point
    def __init__(self, locations, filename=None, flip=False):

        if filename is None:
            self.filename = 'map_test.html'
        if not isinstance(locations, list):
            try:
                self.locations = self.locations.values.tolist()
            except KeyError:
                print('only accept list or dataframe')
                return 0
        else:
            self.locations = locations

        if flip:
            flipped_locations = []

            for location in self.locations:
                new_loc = list(reversed(location[0:2])) #only reverse lat and lon
                new_loc.extend(location[2:])
                flipped_locations.append(new_loc)

            self.location = flipped_locations

        self._build_canvass(self.locations)
        print('the folium map module assumes latitude comes first')
        print('if your data has longitude as the first argument. set flip=True')

    def _build_canvass(self, locations):
        lat_sum = 0; lon_sum = 0
        for row in locations:
            lat_sum = lat_sum + row[0]
            lon_sum = lon_sum + row[1]
        average_lat = lat_sum/len(locations)
        average_lon = lon_sum/len(locations)
        center = [average_lat, average_lon]
        self.canvass = Map(location=center, zoom_start=6)

    def add_point(self, points=None):
        if points is None:
            points = self.locations
        for point in points:
            Marker(location=point[0:2]).add_to(self.canvass)
        return self

    def add_point_clustered(self, points=None):
        if points is None:
            points = self.locations
        marker_cluster = MarkerCluster(icons="dd").add_to(self.canvass)
        for point in points:
            Marker(location = point[0:2]).add_to(marker_cluster)
        return self

    def add_contour(self, contour='SC'):

        if isinstance(contour, str):
             contour_list = get_state_contours(contour)[-1]
             lons, lats= contour_list.exterior.coords.xy
             list_of_coords = list(zip(lats, lons))
        else:
            list_of_coords = contour

        PolygonMarker(list_of_coords, color='blue', fill_opacity=0.2, weight=1).add_to(self.canvass)
        return self

    def add_value(self, values=None, multiplier=4):
        if values is None:
            values = self.locations

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

    def plot(self, open_in_browser=True):
        output_path = os.path.join(os.getcwd(), self.filename)
        self.canvass.save(output_path)
        print(f'the map has been saved to {self.filename}')
        if open_in_browser:
            webbrowser.open('file://' + os.path.realpath(output_path))

if __name__ == '__main__':
    s = SpatialPlotter([[34, -80]])
    s.add_point_clustered([[33.9, -80], [33.8, -80], [33.2, -80]])\
     .plot()


