import os, re, tarfile, requests, fiona, shutil, numpy
import concurrent.futures
from shapely.geometry import Point
from shapely.prepared import prep
from helper_functions import generate_in_between_dates, get_state_contours

class RainfallDownloaderByState:
    '''
    Downloading Precipitation dataset from national weather service (NWS)
    The available dates ranges from 01/01/2005 to 06/27/2017
    Args:
        local_dir (string): the local dir to store data
        start (string): starting date: e.g. '1990-01-01'
        end (string): ending date: e.g. '1990-01-30'

        var_name (str): the name of the variable of interest.
        region (polygon file, optional): the polygon file specifes the area of interest
            default value is south carolina
    '''
    def __init__(self, start, end, local_dir, var_name='GLOBVALUE', state_name='South Carolina',  fill_missing_locs=True):

        self.web_loc = 'https://water.weather.gov/precip/archive'
        self.local_dir = local_dir
        self.start = start
        self.end = end
        self.var_name = var_name #response variable name
        self.var_name_lowercase = self.var_name.lower().capitalize()

        self.fill_missing_locs = fill_missing_locs
        points, _ = get_state_contours(state_name)
        lats = numpy.round(points.LAT.values, 4)
        lons = numpy.round(points.LON.values, 4)
        self.all_points_set = set((x, y) for x, y in zip(lons, lats))

    @staticmethod
    def _generate_io_link(web_url, local_dir, date_token):
        '''
        generate the url for data, and local dir to store the data.
        Args:
            web_url (string): the link for the web archive with rainfall data
            local_dir (string): the dir to store data
            date_token (string): the date of data user wish to retrieve
        Return
            the web url for data source and the local dir to store data
        '''
        year_, month_, date_ = date_token.split('-')
        date_nospace = ''.join([year_, month_, date_])
        web_file_name = f'{year_}/{month_}/{date_}/nws_precip_1day_observed_shape_{date_nospace}.tar.gz'
        local_file_name = date_nospace + '.tar.gz'

        dir_in = os.path.join(web_url, web_file_name)
        dir_out = os.path.join(local_dir, local_file_name)
        return dir_in, dir_out

    def process(self, shp_file):
        observations = []
        # self.region = prep(self.region)
        all_points_set_copy = self.all_points_set.copy()

        shp_file = fiona.open(shp_file)
        for shp_file_entry in shp_file:
            coord = [round(c, 4) for c in list(shp_file_entry['geometry']['coordinates'])]

            #if self.region.contains(Point(coord)): #more accurate but slower
            if tuple(coord) in all_points_set_copy:
                all_points_set_copy.remove(tuple(coord))
                try:
                    coord.append(shp_file_entry['properties'][self.var_name])
                except KeyError:
                    coord.append(shp_file_entry['properties'][self.var_name_lowercase])

                if coord[-1] < 0: #handling missing data
                    coord[-1] = numpy.nan
                observations.append(coord)

        if self.fill_missing_locs:
            non_observations = [list(i) for i in all_points_set_copy]
            for row in non_observations:
                row.append(0)
            observations.extend(non_observations)
        return observations

    def file_download_and_process(self, in_and_out):

        input_link, name_out = in_and_out
        content = requests.get(input_link)
        with open(name_out, 'wb') as f:
            for chunk in content.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        file_size = os.path.getsize(name_out)

        if file_size:
            print(f'finished writing data to file {name_out}')

            dir_, fname = os.path.split(name_out)
            target_folder_name = re.findall('\d{8}', fname)[0]#8-number, should be date

            with tarfile.open(name_out) as tar:
                tar.extractall(os.path.join(dir_, target_folder_name))

            os.remove(name_out) #tear down

            abs_target_folder = os.path.join(dir_, target_folder_name)
            for file in os.listdir(abs_target_folder):
                if file.endswith('.shp'):
                    file_abs = os.path.join(abs_target_folder, file)
                    output_from_process = self.process(file_abs)

                    txt_name = abs_target_folder + '.txt'
                    # output_from_process.to_csv(csv_name)
                    with open(txt_name, 'w') as f:
                        f.write('LONGITUDE LATITUDE PRCP\n')
                        for row in output_from_process:
                            row_string = ' '.join([str(i) for i in row])
                            f.write(f"{row_string}\n")

                    shutil.rmtree(abs_target_folder) #clean up
            return 0
        else:
            raise ValueError('the file is empty. Check the link')
            return None

    def run(self, multiprocess=True):
        '''
        only turn off multiprocess for debugging.
        '''
        job_list = generate_in_between_dates(self.start, self.end)
        in_and_outs = []
        for date_ in job_list:
            link_in, dir_out = RainfallDownloaderByState._generate_io_link(self.web_loc, self.local_dir, date_)
            in_and_outs.append([link_in, dir_out])
        if multiprocess:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=8)
            start_downloading = executor.map(self.file_download_and_process, in_and_outs)
        else:
            print('multiprocessing has been turned off')
            for arg in in_and_outs:
                self.file_download_and_process(arg)

if __name__ == '__main__':
    #example_link
    #https://water.weather.gov/precip/archive/2014/01/01/nws_precip_1day_observed_shape_20140101.tar.gz
    download_handler = RainfallDownloaderByState(start='2014-07-05',
                                                 end='2014-07-05',
                                                 local_dir='./rainfall_data_nc2',
                                                 var_name='GLOBVALUE',
                                                 state_name='South Carolina',
                                                 fill_missing_locs=True).run(False)
