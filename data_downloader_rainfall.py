import os, re, tarfile, requests, fiona, shutil, numpy
import concurrent.futures
from shapely.geometry import Point
from shapely.prepared import prep
from helper_functions import generate_in_between_dates, get_state_contours
from datetime import datetime

class RainfallDownloaderByState:
    '''
    Downloading Precipitation dataset from national weather service (NWS)
    The available dates ranges from 01/01/2005 to 06/27/2017
    This class will download the rainfall information with a 16km^2
    resolution for a given state
    Args:
        local_dir (string): the local dir to store data
        start (string): starting date: e.g. '1990-01-01'
        end (string): ending date: e.g. '1990-01-30'

        var_name (str, optional): the name of the variable of interest. NWS uses globvalue in general
        state_name (string, optional): the name of the state
            default value is 'south carolina'
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
        parsed_date = re.findall('\d{8}', shp_file)[0]
        formated_date = '-'.join([parsed_date[0:4], parsed_date[4:6], parsed_date[6:8]])

        observations = []
        # self.region = prep(self.region)
        all_points_set_copy = self.all_points_set.copy()

        shp_file = fiona.open(shp_file)
        for shp_file_entry in shp_file:
            coord = [round(c, 4) for c in (shp_file_entry['geometry']['coordinates'])]

            #if self.region.contains(Point(coord)): #more accurate but slower
            try:
                all_points_set_copy.remove(tuple(coord))
                try:
                    coord.append(shp_file_entry['properties'][self.var_name])
                except KeyError:
                    coord.append(shp_file_entry['properties'][self.var_name_lowercase])

                if coord[-1] < 0: #handling missing data
                    coord[-1] = numpy.nan

                coord.append(formated_date)
                observations.append(coord)

            except KeyError:
                pass #what remains is location with no observations.

        if self.fill_missing_locs:
            non_observations = [list(i) for i in all_points_set_copy]
            for row in non_observations:
                row.extend([0.0, formated_date])
            observations.extend(non_observations)
        return observations

    def file_download_and_process(self, in_and_out):

        input_link, name_out = in_and_out
        content = requests.get(input_link).content

        with open(name_out, 'wb') as f:
            f.write(content)
            print(f'finished writing data to file {name_out}')
        # if os.path.getsize(name_out):

        dir_, fname = os.path.split(name_out)
        target_folder_name = re.findall('\d{8}', fname)[0]#8-number, should be date

        with tarfile.open(name_out) as tar:
            try:
                tar.extractall(os.path.join(dir_, target_folder_name))
            except EOFError:
                print('the file is corrupt. Unpacking failed.')
                return None

        os.remove(name_out) #tear down
        abs_target_folder = os.path.join(dir_, target_folder_name)
        for file in os.listdir(abs_target_folder):
            if file.endswith('.shp'):
                file_abs = os.path.join(abs_target_folder, file)
                output_from_process = self.process(file_abs)
                txt_name = abs_target_folder + '.txt'
                # output_from_process.to_csv(csv_name)

                with open(txt_name, 'w') as f:
                    for row in output_from_process:
                        row_string = ' '.join([str(i) for i in row])
                        f.write(f'{row_string}\n')
                shutil.rmtree(abs_target_folder) #clean up
        return 0

    def cleaning_up(self, job_list):
        data_ = []
        missing_data = []

        make_name = '-'.join([job_list[0].replace('-', ''), job_list[-1].replace('-', '')])
        target_big_file = os.path.join(self.local_dir, make_name+'.txt')
        with open(target_big_file, 'w') as f:
            f.write('LONGITUDE LATITUDE PRCP DATE\n')
            while job_list:
                date_ = job_list.pop()
                file_ = ''.join([''.join(date_.split('-')), '.txt'])
                file_abs = os.path.join(self.local_dir, file_)
                try:
                    with open(file_abs) as a_single_day:
                        next(a_single_day)
                        f.write(a_single_day.read())
                        os.remove(file_abs)
                except FileNotFoundError:
                    missing_data.append(date_)

        if missing_data:
            print(f'the following dates are missing {missing_data}')

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
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=32)
            start_downloading = executor.map(self.file_download_and_process, in_and_outs)

        else:
            print('multiprocessing has been turned off')
            for arg in in_and_outs:
                self.file_download_and_process(arg)
        self.cleaning_up(job_list)


if __name__ == '__main__':
    #example_link
    #https://water.weather.gov/precip/archive/2014/01/01/nws_precip_1day_observed_shape_20140101.tar.gz
    download_handler = RainfallDownloaderByState(start='2014-07-06',
                                                 end='2014-07-09',
                                                 local_dir='./rainfall_data_nc_exp/',
                                                 var_name='GLOBVALUE',
                                                 state_name='South Carolina',
                                                 fill_missing_locs=True).run(False)
