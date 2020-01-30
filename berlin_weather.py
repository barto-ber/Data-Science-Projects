import pandas as pd
from pandas import read_csv
import matplotlib.pylab as plt
from matplotlib import dates
from datetime import datetime
import numpy as np
import seaborn as sns; sns.set()
import calmap
from openpyxl import Workbook
pd.options.display.width = 0
pd.options.display.max_rows = None
# pd.set_option('display.max_columns', 50)
# pd.set_option('display.width', 1000)

# Reading the weather data txt. Making a copy of df data
data = read_csv('berlin_klima_1948_2018.txt', header=0, parse_dates=[1], sep=';')
data_copy = data.copy()

## Checking the datatypes of columns and other data.

# print(data_copy.dtypes)
# print(type(data_copy['MESS_DATUM']))
# print(data_copy.head())
# print(data_copy.columns)

# Selecting data which I like to work (optional) with.
selected_cols = ['MESS_DATUM', ' TXK', ' TNK', ' TMK']
data_copy = data_copy[selected_cols]

# Renaming some columns
data_copy = data_copy.rename(columns={
							'MESS_DATUM': 'Measurement day',
							' TXK': 'Max daily temp',
							' TNK': 'Min daily temp',
							' TMK': 'Medium daily temp'})
# print(data_copy.columns)
# print(data_copy.head())
# print(type(data_copy))

# # Some Statistics
# print(data_copy.describe())
# temp_max_sort = data_copy.sort_values(by='Max daily temp', ascending=False)
# print(temp_max_sort[:50])
# temp_more30 = data_copy.loc[data_copy['Max daily temp'] > 30]
# sns.relplot(x="Measurement day",
# 			y="Max daily temp",
# 			hue="Max daily temp",
# 			palette="YlOrRd",
# 			data=temp_more30
# 			);
# plt.title("Berlin temperatures over 30°C between 1948 - 2018", size=16)
# plt.ylim(30, 40)

# temp_lessminus10 = data_copy.loc[data_copy['Min daily temp'] < -10]
# sns.relplot(x="Measurement day",
# 			y="Min daily temp",
# 			hue="Min daily temp",
# 			palette="Blues_r",
# 			data=temp_lessminus10
# 			);
# plt.title("Berlin temperatures below -10°C between 1948 - 2018", size=16)
# plt.ylim(-25, -10)


## Converting the date to datetime if not already is.
# data_copy['Measurement day'] = pd.Series.to_numpy(data_copy['Measurement day'])
# print(type(data_copy['Measurement day']))

# # Grouping dates in decades
# data_copy = data_copy.groupby(pd.cut(data_copy['Measurement day'], pd.date_range('1948', '2019', freq='10YS'), right=False)).mean()
# data_copy.reset_index(inplace=True)
# data_copy.reset_index(inplace=True)
# print(data_copy)

## Selecting data based on dates; Setting Measurement day as the index.
# data_copy = data_copy.set_index('Measurement day')
# print(data_copy.head())
# first_jan = data_copy['2016-01-01': '2016-01-31']
# print(first_jan)

## Selecting many periods to put them together on one plot.
# summer1 = data_copy['1970-06-01': '1970-08-31']
# summer2 = data_copy['1980-06-01': '1980-08-31']
# summer3 = data_copy['1990-06-01': '1990-08-31']
# summer4 = data_copy['2000-06-01': '2000-08-31']
# summer5 = data_copy['2010-06-01': '2010-08-31']
# summer6 = data_copy['2018-06-01': '2018-08-31']

## Aggregating data with resample() and datetime index
# monthly = data_copy.resample(rule='M').mean()
# print(monthly.head())

# Basic plots of weather data.
# data_copy.plot(x='Measurement day', y='Max daily temp')
# monthly.plot(y='Max daily temp', kind='line', lw=0.75, c='r')
# plt.xticks(rotation=45)

## Plot of many plots of periods on one plot.
# min_temp = 10
# max_temp = 40
# lw = 1.5
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16,12))
# ax11 = axes[0][0]
# ax12 = axes[0][1]
# ax13 = axes[0][2]
# ax21 = axes[1][0]
# ax22 = axes[1][1]
# ax23 = axes[1][2]
# summer1.plot(y='Max daily temp', ax=ax11, legend=False, lw=lw, ylim=(min_temp, max_temp))
# summer2.plot(y='Max daily temp', ax=ax12, legend=False, lw=lw, ylim=(min_temp, max_temp))
# summer3.plot(y='Max daily temp', ax=ax13, legend=False, lw=lw, ylim=(min_temp, max_temp))
# summer4.plot(y='Max daily temp', ax=ax21, legend=False, lw=lw, ylim=(min_temp, max_temp))
# summer5.plot(y='Max daily temp', ax=ax22, legend=False, lw=lw, ylim=(min_temp, max_temp))
# summer6.plot(y='Max daily temp', ax=ax23, legend=False, lw=lw, ylim=(min_temp, max_temp))

## Changing the ticks of the axis.
# yticks = np.arange(start=10, stop=40, step=5)
# for ax in [ax11, ax12, ax13, ax21, ax22, ax23]:
# 	# Clear x axis ticks.
# 	ax.get_xaxis().set_ticks([])
# 	# Specify y axis ticks.
# 	ax.yaxis.set_ticks(yticks)
# 	# Specify major tick label sizes larger.
# 	ax.tick_params(axis='both', which='major', labelsize=12)
# for ax in [ax11, ax12, ax13, ax21, ax22, ax23]:
# 	# Set minor ticks with day numbers.
# 	ax.xaxis.set_minor_locator(dates.DayLocator(interval=7))
# 	ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))
# 	# Set major ticks with month names.
# 	ax.xaxis.set_major_locator(dates.MonthLocator())
# 	ax.xaxis.set_major_formatter(dates.DateFormatter('\n%b'))

## Adding annotations to each small plot. First defining y und x position.
# all_y = 12
# summer1_x = datetime(1970, 8, 15)
# summer2_x = datetime(1980, 8, 15)
# summer3_x = datetime(1990, 8, 15)
# summer4_x = datetime(2000, 8, 15)
# summer5_x = datetime(2010, 8, 15)
# summer6_x = datetime(2018, 8, 15)

# ax11.text(summer1_x, all_y, '1970', size=16)
# ax12.text(summer2_x, all_y, '1980', size=16)
# ax13.text(summer3_x, all_y, '1990', size=16)
# ax21.text(summer4_x, all_y, '2000', size=16)
# ax22.text(summer5_x, all_y, '2010', size=16)
# ax23.text(summer6_x, all_y, '2018', size=16)

## Adding a big clear plot behind the small plots to add titel and big one name of y axis.
# fig.add_subplot(111, frameon=False);
# plt.grid(False)
# plt.tick_params(labelcolor='none', length=0, width=0, top='off', bottom='off', left='off', right='off')
# plt.ylabel("Temperature in Celsius", size=16, family='Arial')
# plt.title("Summers variations in temperature in Berlin", size=22, family='Arial');
# plt.tight_layout()
# plt.savefig("Summer temps in Berlin.png", transparent=True, dpi=600)
# plt.show()



# # Ploting with seaborn
# ax = sns.lineplot(x=data_copy['Measurement day'].dt.dayofyear,
# 					y=data_copy['Max daily temp'],
# 					hue=data_copy['Measurement day'].dt.year,
# 					legend='full',
# 					palette="seismic",
# 					ci=None,
# 				);
# plt.title("Berlin maximal daily temperatures 1948 - 2018", size=16)
# plt.legend(ncol=5, loc='lower center', fontsize=10)
# plt.xlim(1, 366)
# plt.ylim(-20, 40)
# plt.show()


## CALMAP PLOT
# data_copy['year'] = pd.DatetimeIndex(data_copy['Measurement day']).year
# data_copy['month'] = pd.DatetimeIndex(data_copy['Measurement day']).month
# data_copy['day'] = pd.DatetimeIndex(data_copy['Measurement day']).day

# # Calmap CALENDARplot for one defined year.
# data_copy.set_index('Measurement day', inplace=True)
# fig, ax = calmap.calendarplot(data_copy['1948']['Max daily temp'],
# 					cmap= 'coolwarm',
# 					fig_kws={'figsize': (16,10)},
# 					yearlabel_kws={'color':'black', 'fontsize':24},
# 					subplot_kws={'title':'Berlin max daily temp'},
# 					)
# fig.colorbar(ax[0].get_children()[1], ax=ax.ravel().tolist(), orientation='horizontal')

# # Calmap CALENDARplot for whole years (not visible well cause to many years)
# data_copy.set_index('Measurement day', inplace=True)
# fig, ax = calmap.calendarplot(data_copy['Max daily temp'],
# 					cmap= 'coolwarm',
# 					fig_kws={'figsize': (16,10)},
# 					yearlabel_kws={'color':'black', 'fontsize':24},
# 					subplot_kws={'title':'Berlin max daily temp'}
# 					)
# fig.colorbar(ax[0].get_children()[1], ax=ax.ravel().tolist(), orientation='horizontal')

# # Calmap YEARplot with colorbar.
# fig = plt.figure(figsize=(20,8))
# ax = fig.add_subplot(111)

# cax=calmap.yearplot(data_copy['2018']['Max daily temp'],
# 					cmap= 'coolwarm',
# 					ax=ax
# 					)
# fig.colorbar(cax.get_children()[1], ax=cax, orientation='horizontal')

plt.show()

chosen_temp = 25
data_copy['day_of_week'] = data_copy['Measurement day'].apply(lambda x: x.weekday())
data_copy['weekend'] = data_copy['day_of_week'] >= 5
data_copy['hot'] = data_copy['Max daily temp'] >= chosen_temp
data_copy['go_to_lake_day'] = (data_copy['day_of_week']) & (data_copy['hot'])
data_copy['go_to_lake_weekend'] = (data_copy['weekend']) & (data_copy['hot'])
# print(data_copy[150:200])

# data_copy['year'] = pd.DatetimeIndex(data_copy['Measurement day']).year
# data_copy['month'] = pd.DatetimeIndex(data_copy['Measurement day']).month
# data_copy['day'] = pd.DatetimeIndex(data_copy['Measurement day']).day

lake_any_day = data_copy[data_copy['day_of_week'] >= 0].groupby('Measurement day')['go_to_lake_day'].any()
lake_any_day = pd.DataFrame(lake_any_day).reset_index()
lake_weekend = data_copy[data_copy['weekend'] == True].groupby('Measurement day')['go_to_lake_weekend'].any()
lake_weekend = pd.DataFrame(lake_weekend).reset_index()
# print(lake_any_day[150:200])
lake_any_day['year'] = lake_any_day['Measurement day'].apply(lambda x: x.year)
lake_weekend['year'] = lake_weekend['Measurement day'].apply(lambda x: x.year)

# lake_weekend['month'] = lake_weekend['Measurement day'].apply(lambda x: x.month)

yearly_lake_weekend = lake_weekend.groupby('year')['go_to_lake_weekend'].value_counts().rename('days').reset_index()
yearly_lake_any_day = lake_any_day.groupby('year')['go_to_lake_day'].value_counts().rename('days').reset_index()
# monthly = lake_weekend.groupby('month')['go_to_lake'].value_counts().rename('days').reset_index()
print(yearly_lake_weekend)

# Filtering only lake weekends or/and other days from one defined year.
# First possibility is with lambda the second one with pythonic way for-loop.
# criterion_year = lake_weekend['year'].map(lambda x: x == 2018)
# lake_weekend_2018=lake_weekend[criterion_year]

chosen_year = 2014
lake_weekend_in_year = lake_weekend[[x==chosen_year for x in lake_weekend['year']] & (lake_weekend['go_to_lake_weekend'] == True)]
# lake_weekend_in_year = lake_weekend[criterion_year & (lake_weekend['go_to_lake'] == True)]
# print(f"In your chosen year {chosen_year} were {num_days_hot} hot days with more than {chosen_temp} °C.")

# print(lake_weekend_in_year)

# # Combining into a one excel sheet
# print("### Getting your data")
# print("##### Creating an Excel XLSX file. Please wait!")
# writer = pd.ExcelWriter('ber_rain.xlsx', engine='xlsxwriter')
# data_copy.to_excel(writer, startrow=1, index=False)
# writer.save()
# print("######## The Excel XLSX file has been created.")