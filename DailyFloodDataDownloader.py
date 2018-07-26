import numpy as np
import pandas as pd
import urllib

# example_url = 'https://waterdata.usgs.gov/nwis/dv?cb_00065=on&format=rdb&site_no=02110400&referred_module=sw&period=&begin_date=1990-09-04&end_date=2019-09-04'

class DailyFloodDataDownloader:

    '''
    Get the daily gauge data from the usgs website. The list of
    locations are defined in NWISMapperExport.csv. User can also upload
    a customized file of locations and site numbers.

    Args (summary_file dataframe): Optional. A summary of gage stations where we retrieve data from.
    '''

    def __init__(self, summary_file = None):

        self.url_part_1 = 'https://waterdata.usgs.gov/nwis/dv?cb_00065=on&format=rdb&site_no='
        self.url_part_2 = '&referred_module=sw&period=&begin_date=1800-01-01&end_date=2020-01-01'

        if summary_file is None:
            self.summary_file = pd.read_csv('./data/NWISMapperExport.csv', dtype = {'SiteNumber': str})
            self.station_list = self.summary_file['SiteNumber']

    def _single_station_parser(self, station_id):

        data_content = []
        token = self.url_part_1 + str(station_id) + self.url_part_2

        with urllib.request.urlopen(token) as web_file:
            for line in web_file:
                line = line.decode('utf-8')
                if line.startswith('#'):
                    pass
                else:
                    contents = line.split('\t')
                    if 'USGS' not in contents and len(contents[0]) >= 3:
                        columns = contents
                    else:
                        data_content.append(contents)
        try:
            data_content.pop(0) if '5s' in data_content[0] else 0
        except IndexError:
            print(f'The link is empty. Here is the response from the server:{contents}')
            return None

        retained_vars = [column for column in columns if column.endswith(('00001', '00002', '00003', 'datetime'))]
        flood_df_single_loc = pd.DataFrame(data_content, columns = columns)[retained_vars]

        if not flood_df_single_loc.empty:
            try:
                flood_df_single_loc.columns = ['DATE', 'GAGE_MAX', 'GAGE_MIN', 'GAGE_MEAN']

            except ValueError:
                print(f'got less than four columns remaining. The info of station {station_id} is discarded.')
                return None

            else:
                flood_df_single_loc['LATITUDE'] = float(self.summary_file[self.summary_file.SiteNumber == station_id]['SiteLatitude']) #broadcast!
                flood_df_single_loc['LONGITUDE'] = float(self.summary_file[self.summary_file.SiteNumber == station_id]['SiteLongitude'])
                flood_df_single_loc['SITENUMBER'] = str(station_id)

                print(f'finished processing site number {station_id}')
                return flood_df_single_loc
        else:
            print('Got an empty data frame.')
            return None

    def parsing_engine(self, multiprocessing = True):

        '''
        Args:
            multiprocessing (bool): The parsing engine uses multiprocessing in default. Highly recommended if
            you don't want to spend all day on this.
        '''

        if multiprocessing:
            import concurrent.futures
            executor = concurrent.futures.ProcessPoolExecutor(max_workers = 8)
            result_generator = executor.map(self._single_station_parser, self.station_list)
            all_stations_df = pd.concat([df for df in result_generator])
        else:
            all_stations_df = []
            for station_id in self.station_list:
                single_station_info = self._single_station_parser(station_id)
                all_stations_df.append(single_station_info)
            all_stations_df = pd.concat(all_stations_df, axis = 0)

        number_of_records = all_stations_df.shape[0]
        number_of_locs = len(np.unique(all_stations_df['SITENUMBER']))

        print(f'There are {number_of_locs} locations and {number_of_records} in total.')

        return all_stations_df

if __name__ == '__main__':

    scraper = DailyFloodDataDownloader()
    test = scraper.parsing_engine(multiprocessing = True)
    test.to_csv('/Users/haigangliu/Dropbox/DissertationCode/synthetic_data/flood_data_daily.csv')
    print(test.head())
