import requests, re
import numpy as np
import pandas as pd
import concurrent.futures

class DailyFloodDataDownloader:
    '''
    Get the daily gauge data from the usgs website. The list of
    locations are be accessed in .summary_file.
    Users can also upload a customized file of locations and site numbers by passing a directory to summary_file argument.
    The summary file must have three columns: SiteNumber, SiteLatitude and SiteLongitude

    Args (summary_file dataframe): Optional. A summary of gage stations where we retrieve data from.
    '''
    def __init__(self, summary_file = None):

        self.url_part_1 = 'https://waterdata.usgs.gov/nwis/dv?cb_00065=on&format=rdb&site_no='
        self.url_part_2 = '&referred_module=sw&period=&begin_date=1800-01-01&end_date=2020-01-01'

        if summary_file is None:
            self.summary_file = pd.read_csv('./data/NWISMapperExport.csv', dtype = {'SiteNumber': str})
            self.station_list = self.summary_file['SiteNumber']

    def _single_page_parser(self, station_id):

        '''
        Args: the station id (str)
        return: a dataframe of gage informaion
        '''
        content = requests.get(self.url_part_1 + str(station_id) +
        self.url_part_2)
        obs = []; var_names = None

        if content is None:
            print(f'got an empty page for station {station_id}')
            return None

        for line in content.text.split('\n'):
            is_obs = re.findall("\d+\.\d+", line)
            is_var_name = re.findall('agency', line)

            if is_obs:
                obs.append(line)
            if is_var_name:
                var_names = line

        dataframe = []
        for i in obs:
            cols = i.split('\t')
            dataframe.append(cols)

        if dataframe and var_names:
            output = pd.DataFrame(dataframe, columns=var_names.split('\t'))
            print(f'finished downloading data from station {station_id}')

            output = self._rename_and_clean_up(output, station_id)
            return output
        else:
            print(f'station {station_id} does not have valid record.')
            return None

    def _rename_and_clean_up(self, dataframe, station_id):
        '''
        renaming several variables
        '''
        for column in dataframe.columns:

            if column == 'agency_cd':
                dataframe.drop(columns= [column], axis=1, inplace=True)

            elif column == 'datetime':
                dataframe.rename(columns={column: 'DATE'},inplace=True)

            elif column == 'site_no':
                dataframe.rename(columns={column: 'SITENUMBER'},inplace=True)

            elif column.endswith('_cd'):
                dataframe.drop(columns= [column], axis=1, inplace=True)

            elif column.endswith('_'.join(['00065', '00001'])):
                if 'GAGE_MAX' not in dataframe.columns:
                    dataframe.rename(columns={column: 'GAGE_MAX'}, inplace=True)
                    dataframe['GAGE_MAX'] = pd.to_numeric(dataframe['GAGE_MAX'], errors='coerce')
                else:
                    print('source file might have two duplicate columns for GAGE_MAX. Only first one kept')
                    dataframe.drop(columns= [column], axis=1, inplace=True)

            elif column.endswith('_'.join(['00065', '00002'])):
                if 'GAGE_MIN' not in dataframe.columns:
                    dataframe.rename(columns={column: 'GAGE_MIN'},inplace=True)
                    dataframe['GAGE_MIN'] = pd.to_numeric(dataframe['GAGE_MIN'], errors='coerce')
                else:
                    print('source file might have two duplicate columns for GAGE_MIN. Only first one kept')
                    dataframe.drop(columns= [column], axis=1, inplace=True)

            elif column.endswith('_'.join(['00065', '00003'])):
                if 'GAGE_MEAN' not in dataframe.columns:
                    dataframe.rename(columns={column: 'GAGE_MEAN'},inplace=True)
                    dataframe['GAGE_MEAN'] = pd.to_numeric(dataframe['GAGE_MEAN'], errors='coerce')
                else:
                    print('source file might have two duplicate columns for GAGE_MEAN. Only first one kept')
                    dataframe.drop(columns= [column], axis=1, inplace=True)
            else:
                dataframe.drop(columns= [column], axis=1, inplace=True)

            dataframe['LATITUDE'] = float(self.summary_file[self.summary_file.SiteNumber == station_id]['SiteLatitude']) #broadcast!
            dataframe['LONGITUDE'] = float(self.summary_file[self.summary_file.SiteNumber == station_id]['SiteLongitude'])

        return dataframe

    def run(self, multiprocessing=True):
        '''
        Multiprocessing will be used first. If it fails,
        we'll fall back to single processing.
        '''
        if multiprocessing:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers = 64)
            all_dataframes = executor.map(self._single_page_parser, self.station_list)
            all_stations_df = pd.concat(list(all_dataframes), sort=False)
        else:
            print('multiprocessing module has been turned off')
            all_stations_df = []
            for station_id in self.station_list:
                single_station_info = self._single_page_parser(station_id)
                all_stations_df.append(single_station_info)
            all_stations_df = pd.concat(all_stations_df, axis=0)

        number_of_records = len(all_stations_df)
        number_of_locs = len(np.unique(all_stations_df['SITENUMBER']))

        print(f'There are {number_of_locs} locations and {number_of_records} entries in total.')

        return all_stations_df

if __name__ == '__main__':

    df_downloaded = DailyFloodDataDownloader().run(multiprocessing=True)
    print(df_downloaded.head())
    # df_downloaded.to_csv('./data/flood_data_daily_beta.csv')

    # example_url = 'https://waterdata.usgs.gov/nwis/dv?cb_00065=on&format=rdb&site_no=02110400&referred_module=sw&period=&begin_date=1990-09-04&end_date=2019-09-04'
