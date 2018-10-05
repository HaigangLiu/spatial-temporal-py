import os, re, tarfile, requests, fiona, shutil, numpy, secrets
from datetime import datetime
from multiprocessing import Pool, cpu_count
from helper_functions import generate_in_between_dates, get_state_contours

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
        local_file_name = ''.join([date_nospace, '.tar.gz'])

        dir_in = os.path.join(web_url, web_file_name)
        dir_out = os.path.join(local_dir, local_file_name)
        return dir_in, dir_out

    def process(self, shp_file):
        parsed_date = re.findall('\d{8}', shp_file)[0]
        formated_date = '-'.join([parsed_date[0:4], parsed_date[4:6], parsed_date[6:8]])

        observations = []
        all_points_set_copy = self.all_points_set.copy()

        for shp_file_entry in fiona.open(shp_file):
            coord = [round(c, 4) for c in (shp_file_entry['geometry']['coordinates'])]
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

    def download_and_save(self, in_and_out):
        '''
        download the file from the given url and save to the local directory.
        downloading will be tried for 10 times until give up
        '''
        input_link, name_out = in_and_out

        for i in range(self.maximum_tries):
            try:
                content = requests.get(input_link).content
                with open(name_out, 'wb') as f:
                    f.write(content)
                    dir_, fname = os.path.split(name_out)
                    target_folder_name = re.findall('\d{8}', fname)[0]#8-number, should be date
                with tarfile.open(name_out) as tar:
                    tar.extractall(os.path.join(dir_, target_folder_name))
                print(f'finished processing the data for {target_folder_name}')
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

        os.remove(name_out) #tear down
        abs_target_folder = os.path.join(dir_, target_folder_name)
        for file in os.listdir(abs_target_folder):
            if file.endswith('.shp'):
                file_abs = os.path.join(abs_target_folder, file)
                output_from_process = self.process(file_abs)
                txt_name = abs_target_folder + '.txt'

                with open(txt_name, 'w') as f:
                    for row in output_from_process:
                        row_string = ' '.join([str(i) for i in row])
                        f.write(f'{row_string}\n')
                shutil.rmtree(abs_target_folder) #clean up

    def cleaning_up(self, job_list):
        '''
        combine daily files into one txt file.
        also report if there is any missing data
        '''
        data_ = []; missing_data = []; done = [] #progress tracker
        make_name = '-'.join([job_list[0].replace('-', ''), job_list[-1].replace('-', '')])
        make_name = make_name + secrets.token_hex(3) #it will not overwritten accidentally
        target_big_file = os.path.join(self.local_dir, make_name+'.txt')
        with open(target_big_file, 'w') as f:
            f.write('LONGITUDE LATITUDE PRCP DATE\n')
            while job_list:
                date_ = job_list.pop()
                file_ = ''.join([''.join(date_.split('-')), '.txt'])
                file_abs = os.path.join(self.local_dir, file_)
                try:
                    with open(file_abs) as a_single_day:
                        f.write(a_single_day.read())
                        done.append(date_)
                        os.remove(file_abs)
                except FileNotFoundError:
                    missing_data.append(date_)
        if missing_data:
            print(f'the following dates are missing {missing_data}')
        return done

    def run(self, multiprocess=True):
        '''
        only turn off multiprocess for debugging.
        '''
        job_list = generate_in_between_dates(self.start, self.end)
        in_and_outs = []; done = []#record job status

        for date_ in job_list:
            link_in, dir_out = RainfallDownloaderByState._generate_io_link(self.web_loc, self.local_dir, date_)
            check_file_name = '.'.join([date_.replace('-', ''),'txt'])

            if os.path.isfile(os.path.join(self.local_dir, check_file_name)):
                print(f'daily rainfall info for date: {date_} already there.')
                done.append(date_)
            else:
                in_and_outs.append([link_in, dir_out])

        if multiprocess:
            pool = Pool(processes=cpu_count())
            result = pool.map(self.download_and_save, in_and_outs)
            finshed_task = self.cleaning_up(job_list)
            done.extend(finshed_task)

        else:
            print('multiprocessing has been turned off')
            for arg in in_and_outs:
                self.download_and_save(arg)
            finshed_task = self.cleaning_up(job_list)
            done.extend(finshed_task)

if __name__ == '__main__':
    #example_link
    #https://water.weather.gov/precip/archive/2014/01/01/nws_precip_1day_observed_shape_20140101.tar.gz
    start = '2007-01-01'; end = '2017-12-31'
    full_length = generate_in_between_dates(start, end)
    finished_list = []
    while len(finished_list) != len(full_length):
        finished_list = RainfallDownloaderByState(start=start,
                                                     end=end,
                                                     maximum_tries = 100,
                                                     local_dir='./rainfall_data_nc_exp/',
                                                     var_name='GLOBVALUE',
                                                     state_name='South Carolina',
                                                     fill_missing_locs=True).run(True)
