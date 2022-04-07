import os
import pandas as pd
import ftplib
import codecs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from zipfile import ZipFile
#%%
# Lecture materials were used for the task completion: https://github.com/rolfbecker/MIE_2.02_GeoInfo_WS2020/blob/master/gi0601_DWD_Stations_and_TS_for_TM_soln/gi0601_DWD_Stations_and_TS_for_TM_V002.ipynb
#%%
# Connection Parameters
server = "opendata.dwd.de"
user = "anonymous"
passwd = ""
#%%
# FTP Directory Definition and Station Description Filename Pattern
# The topic of interest.
topic_dir_ftp = "/hourly/precipitation/historical/"
topic_dir = "/hourly/precipitation/historical/task4/"

# This is the search pattern common to ALL station description file names
station_desc_pattern = "_Beschreibung_Stationen.txt"

# Below this directory tree node all climate data are stored.
ftp_climate_data_dir = "/climate_environment/CDC/observations_germany/climate/"
ftp_dir = ftp_climate_data_dir + topic_dir_ftp

#%%
# Local Directories
local_dir = "../data/"

local_ftp_dir = "../data/original/DWD/"  # Local directory to store local ftp data copies, the local data source or input data.
local_ftp_station_dir = local_ftp_dir + topic_dir  # Local directory where local station info is located
local_ftp_ts_dir = local_ftp_dir + topic_dir  # Local directory where time series downloaded from ftp are located

local_generated_dir = "../data/generated/DWD/"  # The generated of derived data in contrast to local_ftp_dir
local_station_dir = local_generated_dir + topic_dir  # Derived station data, i.e. the CSV file
local_ts_appended_dir = local_generated_dir + topic_dir  # Serially appended time series, long data frame for QGIS TimeManager Plugin

#%%

print(local_ftp_dir)
print(local_ftp_station_dir)
print(local_ftp_ts_dir)
print()
print(local_generated_dir)
print(local_station_dir)
print(local_ts_appended_dir)

#%%

os.makedirs(local_ftp_dir, exist_ok=True)  # it does not complain if the dir already exists.
os.makedirs(local_ftp_station_dir, exist_ok=True)
os.makedirs(local_ftp_ts_dir, exist_ok=True)

os.makedirs(local_generated_dir, exist_ok=True)
os.makedirs(local_station_dir, exist_ok=True)
os.makedirs(local_ts_appended_dir, exist_ok=True)
#%%
# FTP Connect
ftp = ftplib.FTP(server)
res = ftp.login(user=user, passwd=passwd)
print(res)
ret = ftp.cwd(".")
#%%
# FTP Grab File Function
def grab_file(ftpfullname, localfullname):
    try:
        ret = ftp.cwd(".")  # A dummy action to chack the connection and to provoke an exception if necessary.
        localfile = open(localfullname, 'wb')
        ftp.retrbinary('RETR ' + ftpfullname, localfile.write, 1024)
        localfile.close()

    except ftplib.error_perm:
        print("FTP ERROR. Operation not permitted. File not found?")

    except ftplib.error_temp:
        print("FTP ERROR. Timeout.")

    except ConnectionAbortedError:
        print("FTP ERROR. Connection aborted.")
#%%
# Generate Pandas Dataframe from FTP Directory Listing
def gen_df_from_ftp_dir_listing(ftp, ftpdir):
    lines = []
    flist = []
    try:
        res = ftp.retrlines("LIST " + ftpdir, lines.append)
    except:
        print("Error: ftp.retrlines() failed. ftp timeout? Reconnect!")
        return

    if len(lines) == 0:
        print("Error: ftp dir is empty")
        return

    for line in lines:
        [ftype, fsize, fname] = [line[0:1], int(line[31:42]), line[56:]]
        fext = os.path.splitext(fname)[-1]

        if fext == ".zip":
            station_id = int(fname.split("_")[2])
        else:
            station_id = -1

        flist.append([station_id, fname, fext, fsize, ftype])

    df_ftpdir = pd.DataFrame(flist, columns=["station_id", "name", "ext", "size", "type"])
    return (df_ftpdir)

#%%

df_ftpdir = gen_df_from_ftp_dir_listing(ftp, ftp_dir)
#%%
# Dataframe with TS Zip Files
df_zips = df_ftpdir[df_ftpdir["ext"] == ".zip"]
df_zips.set_index("station_id", inplace=True)  # list of files containing ts
df_zips.head(10)
#%%
# Retrieve the Station Description File Name
station_fname = df_ftpdir[df_ftpdir['name'].str.contains(station_desc_pattern)]["name"].values[0]
print(station_fname)
#%%
# Download the Station Description File
print("grabFile: ")
print("From: " + ftp_dir + station_fname)
print("To:   " + local_ftp_station_dir + station_fname)
grab_file(ftp_dir + station_fname, local_ftp_station_dir + station_fname)
#%%
# extract column names. They are in German (de)
# We have to use codecs because of difficulties with character encoding (German Umlaute)
def station_desc_txt_to_csv(txtfile, csvfile):
    file = codecs.open(txtfile, "r", "utf-8")
    r = file.readline()
    file.close()
    colnames_de = r.split()
    colnames_de

    translate = \
        {'Stations_id': 'station_id',
         'von_datum': 'date_from',
         'bis_datum': 'date_to',
         'Stationshoehe': 'altitude',
         'geoBreite': 'latitude',
         'geoLaenge': 'longitude',
         'Stationsname': 'name',
         'Bundesland': 'state'}

    colnames_en = [translate[h] for h in colnames_de]

    # Skip the first two rows and set the column names.
    df = pd.read_fwf(txtfile, skiprows=2, names=colnames_en, parse_dates=["date_from", "date_to"], index_col=0, encoding='windows-1252')

    # write csv
    df.to_csv(csvfile, sep=";")
    return (df)
#%%
# Convert Station Description File.txt to Station Description File.csv
basename = os.path.splitext(station_fname)[0]
df_stations = station_desc_txt_to_csv(local_ftp_station_dir + station_fname, local_station_dir + basename + ".csv")
df_stations.head()
#%%
# Select Stations Located in NRW from Station Description Dataframe
station_ids_selected = df_stations[df_stations['state'].str.contains("Nordrhein")].index

# Create variable with TRUE if state is Nordrhein-Westfalen
isNRW = df_stations['state'].str.contains("Nordrhein")

# Create variable with TRUE if date_to is latest date (indicates operation up to now)
#isOperational = df_stations['date_to'] == df_stations.date_to.max()
isOperational = df_stations['date_to'] >= datetime(2018, 1, 1)

# select on both conditions
dfNRW_stations = df_stations[isNRW & isOperational]

# Select only data associated with OE and HSK stations
interested_stations_indices = (dfNRW_stations.index == 216) | (dfNRW_stations.index == 2947) | \
                              (dfNRW_stations.index == 1300) | (dfNRW_stations.index == 2483) | \
                              (dfNRW_stations.index == 3215) | (dfNRW_stations.index == 4488) | \
                              (dfNRW_stations.index == 6264) | (dfNRW_stations.index == 5468) | \
                              (dfNRW_stations.index == 7330)

df_stations_interested = dfNRW_stations[interested_stations_indices]
#%%

df_stations_interested.shape
#print("Number of stations in NRW: \n", dfNRW.count())
#%%
# Download TS Data from FTP Server
# Add the names of the zip files only to a list.
def upload_selected_zips(df_selected_stations):
    local_zip_list = []
    station_ids_selected = list(df_selected_stations.index)

    for station_id in station_ids_selected:
        try:
            fname = df_zips["name"][station_id]
            print(fname)
            grab_file(ftp_dir + fname, local_ftp_ts_dir + fname)
            local_zip_list.append(fname)
        except:
            print("WARNING: TS file for key %d not found in FTP directory." % station_id)
    return local_zip_list
#%%
# Number of obtained stations
local_zip_list_interested_stations = upload_selected_zips(df_stations_interested)
len(local_zip_list_interested_stations)
# %%
# Join (Merge) the Time Series Columns
def prec_ts_to_df(fname):
    dateparse = lambda dates: [datetime.strptime(str(d), '%Y%m%d%H') for d in dates]

    df = pd.read_csv(fname, delimiter=";", encoding="utf8", index_col="MESS_DATUM", parse_dates=["MESS_DATUM"],
                     date_parser=dateparse, na_values=[-999.0, -999])

    # df = pd.read_csv(fname, delimiter=";", encoding="iso8859_2",\
    #             index_col="MESS_DATUM", parse_dates = ["MESS_DATUM"], date_parser = dateparse)

    # https://medium.com/@chaimgluck1/working-with-pandas-fixing-messy-column-names-42a54a6659cd

    # Column headers: remove leading blanks (strip), replace " " with "_", and convert to lower case.
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    df.index.name = df.index.name.strip().lower().replace(' ', '_').replace('(', '').replace(')', '')
    return (df)
#%%
# Select hourly precipitation data from TS and join it with each other by station_id
# PRECIPITATION
def prec_ts_merge(local_zip_list):
    # Very compact code.
    df = pd.DataFrame()
    for elt in local_zip_list:
        ffname = local_ftp_ts_dir + elt
        print("Zip archive: " + ffname)
        with ZipFile(ffname) as myzip:
            # read the time series data from the file starting with "produkt"
            prodfilename = [elt for elt in myzip.namelist() if elt.split("_")[0]=="produkt"][0]
            print("Extract product file: %s" % prodfilename)
            print()
            with myzip.open(prodfilename) as myfile:
                dftmp = prec_ts_to_df(myfile)
                s = dftmp["r1"].rename(dftmp["stations_id"][0]).to_frame()
                # outer merge.
                df = pd.merge(df, s, left_index=True, right_index=True, how='outer')

    #df.index.names = ["year"]
    df.index.rename(name="time", inplace = True)
    return(df)
#%%

df_merged_ts = prec_ts_merge(local_zip_list_interested_stations)
#%%

df_merged_ts.shape
#%%
# Select indices with the time period of interest
idx = (df_merged_ts.index >= '2018-04-16 00:00:00') & (df_merged_ts.index < '2018-08-16 01:00:00')
#%%
# Select rows with the time period of interest, get rid of unnecessary columns
df_merged_ts_ex4 = df_merged_ts[idx]
#%%
# Convert hourly data to daily
df_ts_daily = df_merged_ts_ex4.groupby(pd.Grouper(freq='D')).sum()
#%%

df_ts_daily.shape
#%%
# Create separate dfs for Olpe (OE) and Hochsauerlandkreis (HSK) counties
df_ts_oe_daily = df_ts_daily[[216, 2947, 5468]]
df_ts_hsk_daily = df_ts_daily[[1300, 2483, 3215, 4488, 6264, 7330]]
#%%
# Calculate average precipitation rate for Olpe (OE) and Hochsauerlandkreis (HSK) counties
df_ts_oe_daily_average = df_ts_oe_daily.mean(axis=1).to_frame(name='daily_precp')
df_ts_hsk_daily_average = df_ts_hsk_daily.mean(axis=1).to_frame(name='daily_precp')
#%%
# Plot average precipitation rate for Olpe (OE) and Hochsauerlandkreis (HSK) counties
# Used materials: https://scentellegher.github.io/programming/2017/05/24/pandas-bar-plot-with-formatted-dates.html ,
# https://unidata.github.io/python-training/workshop/Time_Series/basic-time-series-plotting/

def plot_ts_average_bars_line(df, county_name, graph_colors):
    df['cumsum_daily_precp'] = df['daily_precp'].cumsum(axis=0)

    # set ggplot style
    plt.style.use('ggplot')
    font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 15}
    plt.rc('font', **font)
    #plot data
    fig, ax = plt.subplots(figsize=(20, 10))
    axb = ax.twinx()

    ax.set_title('Average Precipitation Data of ' + county_name +
                 ' County from 16 April 2018 to 16 August 2018')
    ax.set_ylabel('Precipitation (mm/day)')
    ax.set_xlabel('Date')
    ax.set_ylim([0, 50])

    ax.bar(df.index, df['daily_precp'], linewidth=1.5, color=graph_colors[0], edgecolor=graph_colors[1],
           label='Daily Precipitation')

    axb.set_ylabel('Cumulative Precipitation (mm)')
    axb.plot(df.index, df['cumsum_daily_precp'], linewidth=3.0, color=graph_colors[2],
             label='Cumulative Precipitation')
    axb.set_ylim([0, 250])
    axb.grid(None)

    #set ticks every week
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    #format date
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    #ax.set_xlim([date(2018, 4, 16), date(2018, 8, 16)])

    # Handling of getting lines and labels from all axes for a single legend
    bars, labels = ax.get_legend_handles_labels()
    lines2, labels2 = axb.get_legend_handles_labels()
    axb.legend(bars + lines2, labels + labels2, loc='upper left')

    plt.show()

    return df


df_ts_oe_daily_average_cumsum = plot_ts_average_bars_line(df_ts_oe_daily_average, 'OE',
                                                          graph_colors=['salmon', 'red', 'maroon'])
df_ts_hsk_daily_average_cumsum = plot_ts_average_bars_line(df_ts_hsk_daily_average, 'HSK',
                                                           graph_colors=['mediumaquamarine', 'teal', 'darkslategray'])
#%%
# Plot precipitation rate (for each station) for Olpe (OE) and Hochsauerlandkreis (HSK) counties

def plot_ts_bars(df, county_name):
    plt.style.use('ggplot')
    font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 20}
    plt.rc('font', **font)

    columns_num = int(len(df.columns) / 3)

    fig, ax = plt.subplots(3, columns_num, sharey='row', figsize=(27, 20))
    ax = ax.flatten()

    cmap = plt.get_cmap('tab10')
    colors = iter(cmap(np.arange(cmap.N)))

    for idx, column_name in enumerate(df.columns):
        ax[idx].bar(df.index, df[column_name], color=next(colors))
        # set ticks every week
        ax[idx].xaxis.set_major_locator(mdates.WeekdayLocator())
        # format date
        ax[idx].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax[idx].tick_params(axis='x', rotation=45)
        ax[idx].set_title('Station ID: ' + str(column_name), fontsize=30)
        ax[idx].yaxis.set_tick_params(labelleft=True)
        ax[idx].set_ylabel('Precipitation (mm/day)')
        ax[idx].set_xlabel('Date')
        ax[idx].set_ylim([0, 80])

    fig.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.suptitle('Daily Precipitation of ' + county_name +
                 ' County from 16 April 2018 to 16 August 2018', fontsize=40)

    plt.show()


plot_ts_bars(df_ts_oe_daily, 'OE')
plot_ts_bars(df_ts_hsk_daily, 'HSK')
#%%
# Plot ts of the both counties in one graph
def plot_subplots_bars(ax, dict_idxs, df, county_name):
    cmap = plt.get_cmap('tab10')
    colors = iter(cmap(np.arange(cmap.N)))

    for idx, column_name in dict_idxs.items():
        ax[idx].bar(df.index, df[column_name], color=next(colors))
        # set ticks every week
        ax[idx].xaxis.set_major_locator(mdates.WeekdayLocator())
        # format date
        ax[idx].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax[idx].tick_params(axis='x', rotation=45)
        ax[idx].yaxis.set_tick_params(labelleft=True)
        ax[idx].set_title(county_name + ' - Station ID: ' + str(column_name), fontsize=30)
        ax[idx].set_ylabel('Precipitation (mm/day)')
        ax[idx].set_xlabel('Date')

def plot_ts_all_counties_bars():
    plt.style.use('ggplot')
    font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 23}
    plt.rc('font', **font)

    fig, ax = plt.subplots(3, 3, sharey='all', figsize=(32, 27))
    ax = ax.flatten()

    oe_idxs = range(len(df_ts_oe_daily.columns))
    dict_oe_idxs = dict(zip(oe_idxs, df_ts_oe_daily.columns))

    hsk_idxs = range(len(df_ts_oe_daily.columns), len(df_ts_oe_daily.columns) + len(df_ts_hsk_daily.columns))
    dict_hsk_idxs = dict(zip(hsk_idxs, df_ts_hsk_daily.columns))

    plot_subplots_bars(ax, dict_oe_idxs, df_ts_oe_daily, 'OE')
    plot_subplots_bars(ax, dict_hsk_idxs, df_ts_hsk_daily, 'HSK')

    fig.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.suptitle('Daily Precipitation of OE and HSK from 16 April 2018 to 16 August 2018', fontsize=45)

    plt.show()

plot_ts_all_counties_bars()
#%%
# Plot cumulative precipitation of the both counties in one graph
def plot_subplots_lines(ax, dict_idxs, df, county_name, colors):

    for idx, column_name in dict_idxs.items():
        ax[idx].plot(df.index, df[column_name], color=next(colors), linewidth=3.0)
        # set ticks every week
        ax[idx].xaxis.set_major_locator(mdates.WeekdayLocator())
        # format date
        ax[idx].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax[idx].tick_params(axis='x', rotation=45)
        ax[idx].yaxis.set_tick_params(labelleft=True)
        ax[idx].set_title(county_name + ' - Station ID: ' + str(column_name), fontsize=30)
        ax[idx].set_ylabel('Precipitation (mm)')
        ax[idx].set_xlabel('Date')


def plot_ts_all_counties_lines():
    plt.style.use('ggplot')
    font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 23}
    plt.rc('font', **font)

    fig, ax = plt.subplots(3, 3, sharey='all', figsize=(32, 27))
    ax = ax.flatten()

    cmap = plt.get_cmap('tab10')
    colors = iter(cmap(np.arange(cmap.N)))

    df_oe_cum_sum = df_ts_oe_daily.cumsum(axis=0)
    df_hsk_cum_sum = df_ts_hsk_daily.cumsum(axis=0)

    oe_idxs = range(len(df_oe_cum_sum.columns))
    dict_oe_idxs = dict(zip(oe_idxs, df_oe_cum_sum.columns))

    hsk_idxs = range(len(df_oe_cum_sum.columns), len(df_oe_cum_sum.columns) + len(df_hsk_cum_sum.columns))
    dict_hsk_idxs = dict(zip(hsk_idxs, df_hsk_cum_sum.columns))

    plot_subplots_lines(ax, dict_oe_idxs, df_oe_cum_sum, 'OE', colors)
    plot_subplots_lines(ax, dict_hsk_idxs, df_hsk_cum_sum, 'HSK', colors)

    fig.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.suptitle('Cumulative Precipitation of OE and HSK from 16 April 2018 to 16 August 2018', fontsize=45)

    plt.show()

plot_ts_all_counties_lines()
#%%
# Plot cumulative precipitation (for each station) for Olpe (OE) and Hochsauerlandkreis (HSK) counties

#df_ts_oe_daily_average['cumsum_daily_precp'] = df_ts_oe_daily_average['daily_precp'].cumsum(axis=0)

def plot_ts_cumsum_lines(df, county_name):

    df_cum_sum = df.cumsum(axis=0)

    plt.style.use('ggplot')
    font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 15}
    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(20, 10))

    cmap = plt.get_cmap('tab10')
    colors = iter(cmap(np.arange(cmap.N)))

    for column_name in df_cum_sum.columns:
        x = df_cum_sum.index
        y = df_cum_sum[column_name]
        ax.plot(x, y, color=next(colors), label='Station ID: ' + str(column_name), linewidth=3.0)

        # set ticks every week
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    # format date
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.set_title('Cumulative Precipitation of ' + county_name +
                     ' County from 16 April 2018 to 16 August 2018')
    ax.set_ylabel('Cumulative Precipitation (mm)')
    ax.set_xlabel('Date')
    ax.set_ylim([0, 300])
    ax.legend()
    plt.show()

    return df_cum_sum


df_ts_oe_cum_sum = plot_ts_cumsum_lines(df_ts_oe_daily, 'OE')
df_ts_hsk_cum_sum = plot_ts_cumsum_lines(df_ts_hsk_daily, 'HSK')
#%%
# Save generated dfs (that were used for plotting graphs) to csv
def save_dfs_to_csv(df, csv_name):
    filepathname = local_ts_appended_dir + csv_name
    print(csv_name + ' saved to: %s' % (filepathname))
    df.to_csv(filepathname, sep=";")


save_dfs_to_csv(df_ts_oe_daily, 'ts_oe_daily.csv')
save_dfs_to_csv(df_ts_hsk_daily, 'ts_hsk_daily.csv')

save_dfs_to_csv(df_ts_oe_cum_sum, 'ts_oe_cum_sum.csv')
save_dfs_to_csv(df_ts_hsk_cum_sum, 'ts_hsk_cum_sum.csv')

save_dfs_to_csv(df_ts_oe_daily_average_cumsum, 'ts_oe_daily_average_cumsum.csv')
save_dfs_to_csv(df_ts_hsk_daily_average_cumsum, 'ts_hsk_daily_average_cumsum.csv')

