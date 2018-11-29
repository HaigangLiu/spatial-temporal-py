import os
import re
import tarfile
import requests
import fiona
import shutil
import numpy
import secrets
from multiprocessing import Pool, cpu_count
from utility_functions import get_in_between_dates, get_state_grid_points

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
        maximum_tries (int): specifies how many time we try to dowload the data file until give up
            default value is 10.
        fill_missing_locs (boolean): whether fill with missing locations with 0 if no observations reported on that day. Defaul is True.
    '''
    def __init__(self, start, end, local_dir, var_name='GLOBVALUE', state_name='South Carolina',  maximum_tries=10, fill_missing_locs=True):

        self.web_loc = 'https://water.weather.gov/precip/archive'
        self.local_dir = local_dir
        self.start = start
        self.end = end
        self.var_name = var_name #response variable name
        self.var_name_lowercase = self.var_name.lower().capitalize()
        self.maximum_tries = maximum_tries
        self.fill_missing_locs = fill_missing_locs

        points = get_state_grid_points(state_name)
        lats = numpy.round(points.LAT.values, 4)
        lons = numpy.round(points.LON.values, 4)
        self.all_points_set = set((x, y) for x, y in zip(lons, lats))

    def download(self, web_url, local_file):
        '''
        download the file from the given url and save to the local directory.
        downloading will be tried for 10 times until give up
        '''
        for i in range(self.maximum_tries):
            try:
                content = requests.get(web_url).content
                with open(local_file, 'wb') as f:
                    f.write(content)
                    dir_, fname = os.path.split(local_file)
                    target_folder_name = re.findall('\d{8}', fname)[0]#8-number, should be date
                with tarfile.open(local_file) as tar:
                    tar.extractall(os.path.join(dir_, target_folder_name))
                print(f'unpacking the data for {target_folder_name}')
            except EOFError:
                print(f'connection failed (incomplete tarfile), trying for the {i+2}th time')
                continue
            except tarfile.ReadError:
                print(f'connection failed (empty tarfile), trying for the {i+2}th time')
                continue
            else:
                break
        else:
            print('the file is corrupt. Unpacking failed.')
            return None
        os.remove(local_file) #tear down
        abs_target_folder = os.path.join(dir_, target_folder_name)
        return abs_target_folder

    def process(self, shp_file):
        parsed_date = re.findall('\d{8}', shp_file)[0]
        output_file_name = os.path.join(self.local_dir, '.'.join([parsed_date, 'txt']))
        formated_date = '-'.join([parsed_date[0:4], parsed_date[4:6], parsed_date[6:8]])

        all_points_set_copy = self.all_points_set.copy()

        with open(output_file_name, 'w') as f:
            for shp_file_entry in fiona.open(shp_file):
                coord = [round(c, 4) for c in shp_file_entry['geometry']['coordinates']]
                try:
                    all_points_set_copy.remove(tuple(coord))
                    try:
                        coord.append(shp_file_entry['properties'][self.var_name])
                    except KeyError: #some of has lower case names
                        coord.append(shp_file_entry['properties'][self.var_name_lowercase])
                    if coord[-1] < 0: #handling missing data
                        coord[-1] = numpy.nan
                    coord.append(formated_date)
                    string_to_write = ' '.join([str(i) for i in coord])
                    f.write(f'{string_to_write}\n')

                except KeyError:
                    pass #what remains is location with no observations.

            if self.fill_missing_locs:
                non_observations = [list(i) for i in all_points_set_copy]
                for coord in non_observations:
                    coord.extend([0.0, formated_date])
                    string_to_write = ' '.join([str(i) for i in coord])
                    f.write(f'{string_to_write}\n')
        shutil.rmtree(shp_file)
        return output_file_name

    def merge(self, job_list, coerce=False):
        '''
        combine daily files into one txt file.
        also report if there is any missing data
        '''
        data_ = []; missing_data = []; done = [] #progress tracker
        make_name = '-'.join([job_list[0].replace('-', ''), job_list[-1].replace('-', '')])
        make_name = '-'.join([make_name, secrets.token_hex(2)]) #it will not overwritten accidentally

        target_file = os.path.join(self.local_dir, make_name+'.txt')
        job_list_txt = [job.replace('-','') + '.txt' for job in job_list]
        failed_tasks = [i for i in job_list_txt if i not in os.listdir(self.local_dir)]

        list_of_small_files = []
        for job in job_list:
            txt = '.'.join([job.replace('-', ''), 'txt'])
            abs_path = os.path.join(self.local_dir, txt)
            list_of_small_files.append(abs_path)

        if len(job_list) == 1:
            print('merge function not applied since there is only a single file')
            return 0

        elif len(failed_tasks) > 0 and coerce == False:
            print('merge function not applied since the following jobs have failed')
            print('if you want to merge anyway, set coerce=True to override this setting')
            print(f'{failed_tasks}')

        else:
            with open(target_file, 'w') as f:
                f.write('LONGITUDE LATITUDE PRCP DATE\n')
                while list_of_small_files:
                    file_ = list_of_small_files.pop()
                    with open(file_) as small_file:
                            f.write(small_file.read())
                    os.remove(file_)
            print(f'all files have been merged into a single file {target_file}')

    def run_single_date(self, date_string):
        datestrp = date_string.replace('-', '')
        web_file_template = f'{datestrp[0:4]}/{datestrp[4:6]}/{datestrp[6:8]}/nws_precip_1day_observed_shape_{datestrp}.tar.gz'
        local_file_name = ''.join([datestrp, '.tar.gz'])

        web_url = os.path.join(self.web_loc, web_file_template)
        dir_out = os.path.join(self.local_dir, local_file_name)

        check_file_name = '.'.join([date_string.replace('-', ''),'txt'])

        if os.path.isfile(os.path.join(self.local_dir, check_file_name)):
            print(f'daily rainfall info for date: {date_string} already there.')
        else:
            downloaded_file_link = self.download(web_url, dir_out)
            if downloaded_file_link is not None:
                txt_file = self.process(downloaded_file_link)
            else:
                print(f'processing for {date_string} has failed')

    def run(self, multiprocess=True):
        '''
        only turn off multiprocess for debugging.
        first generate the proper web url to grab data
        and the local link to store the data file
        '''
        job_list = get_in_between_dates(self.start, self.end)

        if multiprocess:
            pool = Pool(processes=cpu_count())
            pool.map(self.run_single_date, job_list)
        else:
            for job in job_list:
                run_single_date(job)
        report = self.merge(job_list)
        return report

if __name__ == '__main__':
    #example_link
    #https://water.weather.gov/precip/archive/2014/01/01/nws_precip_1day_observed_shape_20140101.tar.gz
    start = '2005-01-01'; end = '2017-06-27' #takes 6497.7 sec
    full_length = get_in_between_dates(start, end)
    finished_list = []
    finished_list = RainfallDownloaderByState(start=start,
                                              end=end,
                                              maximum_tries=10,
                                              local_dir='./demo/',
                                              var_name='GLOBVALUE',
                                              state_name='New York',
                                              fill_missing_locs=True).run(True)
