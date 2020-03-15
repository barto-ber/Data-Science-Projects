import pandas as pd
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.pylab as plt
from matplotlib import dates
from datetime import datetime
import numpy as np
import seaborn as sns; sns.set()
from mpl_toolkits.mplot3d import Axes3D
import calmap
from openpyxl import Workbook
pd.options.display.width = 0
pd.options.display.max_rows = None
# pd.set_option('display.max_columns', 50)
# pd.set_option('display.width', 1000)

def read_file():
	# Reading the weather data txt. Making a copy of df data
	data = read_csv('berlin_klima_1948_2019.txt', header=0, parse_dates=[1], sep=';')
	data_copy = data.copy()
	print("--- Data copy: DONE ---")
	print("+++ Check data with def check_data() if needed +++")
	return data_copy

def read_file_max_1876():
	data2 = read_csv('berlin_max_temp_daily_1876_1962.csv',
					 usecols=["Zeitstempel", "Wert"],
					 header=0, parse_dates=["Zeitstempel"], sep=',').rename(
		columns={
			"Zeitstempel": "Measurement day",
			"Wert": "Max daily temp"
		}
	)
	data_copy2 = data2.copy()
	# print(data_copy2.info())
	# print(data_copy2.head())
	data_copy2 = data_copy2.set_index('Measurement day')
	until_1948_max = data_copy2['1876-01-01': '1947-12-31']
	until_1948_max.reset_index(inplace=True)
	# print(until_1948.head())
	# print(until_1948.tail())
	return until_1948_max

def read_file_min_1876():
	data3 = read_csv('berlin_min_temp_daily_1876_1962.csv',
					 usecols=["Zeitstempel", "Wert"],
					 header=0, parse_dates=["Zeitstempel"], sep=',').rename(
		columns={
			"Zeitstempel": "Measurement day",
			"Wert": "Min daily temp"
		}
	)
	data_copy3 = data3.copy()
	# print(data_copy3.info())
	# print(data_copy3.head())
	data_copy3 = data_copy3.set_index('Measurement day')
	until_1948_min = data_copy3['1876-01-01': '1947-12-31']
	until_1948_min.reset_index(inplace=True)
	# print(until_1948_min.head())
	# print(until_1948_min.tail())
	return until_1948_min

def read_file_mean_1876():
	data4 = read_csv('berlin_mean_temp_daily_1876_1962.csv',
					 usecols=["Zeitstempel", "Wert"],
					 header=0, parse_dates=["Zeitstempel"], sep=',').rename(
		columns={
			"Zeitstempel": "Measurement day",
			"Wert": "Medium daily temp"
		}
	)
	data_copy4 = data4.copy()
	# print(data_copy4.info())
	# print(data_copy4.head())
	data_copy4 = data_copy4.set_index('Measurement day')
	until_1948_mean = data_copy4['1876-01-01': '1947-12-31']
	until_1948_mean.reset_index(inplace=True)
	# print(until_1948_mean.head())
	# print(until_1948_mean.tail())
	return until_1948_mean

def check_data():
	# Checking the data types etc.
	data_copy = read_file()
	print(data_copy.shape)
	print(data_copy.dtypes)
	print(type(data_copy['MESS_DATUM']))
	print(data_copy.head())
	print(data_copy.tail())
	print(data_copy.columns)
	print("--- Checking data: DONE ---")

def select_data():
	# Selecting data which I like to work with.
	data_copy = read_file()
	selected_cols = ['MESS_DATUM', ' TXK', ' TNK', ' TMK']
	data_copy = data_copy[selected_cols]
	print("--- Selecting columns: DONE ---")
	# Renaming some columns
	data_copy = data_copy.rename(columns={
								'MESS_DATUM': 'Measurement day',
								' TXK': 'Max daily temp',
								' TNK': 'Min daily temp',
								' TMK': 'Medium daily temp'})
	print("--- Renaming columns: DONE ---")
	data_copy = data_copy.set_index('Measurement day')
	data_copy.reset_index(inplace=True)
	# print(data_copy.columns)
	# print(data_copy.head())
	# print(type(data_copy))
	return data_copy

def combining_datasets():
	data_copy = select_data()
	until_1948_max = read_file_max_1876()
	until_1948_min = read_file_min_1876()
	until_1948_mean = read_file_mean_1876()
	until_1948_max_min = pd.merge(until_1948_max, until_1948_min, on="Measurement day")
	until_1948 = pd.merge(until_1948_max_min, until_1948_mean, on="Measurement day")
	# print(until_1948.head())
	frames = [until_1948, data_copy]
	data_copy_combined = pd.concat(frames, sort=False)
	print("The shape of combined data is:\n", (data_copy_combined.shape))
	# print(data_copy_combined.head())
	# print(data_copy_combined.tail())
	return data_copy_combined

def create_combined_csv():
	data_for_csv = combining_datasets()
	data_for_csv.to_csv("ber_weather_combined_1876_2019.csv", index=False)
	print("--- Combined data copied to CSV ---")

def read_csv_combined():
	data_combined = read_csv('ber_weather_combined_1876_2019.csv', header=0, parse_dates=[0])
	print("--- Combined data read ---\n")
	# print(data_combined.head())
	return data_combined

def groupby_exp():
	g_data = read_csv_combined()
	for temp, group in g_data.groupby('Max daily temp'):
		print(temp)
		print(group)

def matrix_plt():
	m_data = read_csv_combined()
	pd.plotting.scatter_matrix(m_data[['Max daily temp', 'Min daily temp', 'Medium daily temp']])
	plt.show()

def today_before():
	tdata = read_csv_combined()
	tdata['year'] = pd.DatetimeIndex(tdata['Measurement day']).year
	tdata['month'] = pd.DatetimeIndex(tdata['Measurement day']).month
	tdata['day'] = pd.DatetimeIndex(tdata['Measurement day']).day
	# print("Today before data:\n", tdata.head())
	# print(tdata.dtypes)
	x_day = 15
	x_month = 3
	x_year = 1876
	check_today_before = tdata[#(tdata['day'] == x_day) &
							   (tdata['year'] >= x_year) &
							   (tdata['month'] == x_month) #&
							   # (tdata['Max daily temp'] >= 13)
	]
	print(f"\nToday temperature {x_day}.{x_month}.2020 in years before was:\n", check_today_before)
	check_mean = check_today_before.groupby('year')['Max daily temp'].mean()
	print(f"The mean temperatures for month {x_month} from all years are:\n", check_mean)
	# Lets show it on a graph
	sns.catplot(
		x='year',
		y='Max daily temp',
		data=check_today_before,
		ci=None,
		kind='bar',
		color='navy'
	)
	plt.title(f"Berlin: today temperature {x_day}.{x_month}.2020 in the years before", size=14)
	plt.xlabel("Year", size=11)
	plt.ylabel("Max daily temp C°", size=11)
	plt.xticks(rotation=90, size=8)
	plt.yticks(size=8, ticks=(np.arange(-5, 18, 1)))
	plt.show()
	return check_today_before


def months_3d(): # not working properly
	d_month_3d = today_before()
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_trisurf(d_month_3d['Measurement day'].dt.year,
					d_month_3d['Measurement day'].dt.dayofyear,
					d_month_3d['Max daily temp'],
					cmap=plt.cm.jet, linewidth=0.2)
	# to Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)
	ax.view_init(30, 45)
	plt.title("Berlin maximal daily temperatures 1876 - 2019, 3d", size=13)
	plt.show()


def some_statistics():
	data_copy_combined = read_csv_combined()
	print("\n--- Some statistics from the .describe() method ---")
	print(data_copy_combined['Max daily temp'].describe())
	temp_max_sort = data_copy_combined.sort_values(by='Max daily temp', ascending=False)
	print("\n--- Highest temperatures sorted from the highest ---")
	print(temp_max_sort[:50])

	# Plot days with temps over 30°C.
	temp_more30 = data_copy_combined.loc[data_copy_combined['Max daily temp'] > 30]
	sns.relplot(x="Measurement day",
				y="Max daily temp",
				hue="Max daily temp",
				palette="YlOrRd",
				data=temp_more30
				);
	plt.title("Berlin temperatures over 30°C between 1876 - 2019", size=16)
	plt.ylim(30, 40)
	# Plot days with temps less 10°C.
	temp_lessminus10 = data_copy_combined.loc[data_copy_combined['Min daily temp'] < -10]
	sns.relplot(x="Measurement day",
				y="Min daily temp",
				hue="Min daily temp",
				palette="Blues_r",
				data=temp_lessminus10
				);
	plt.title("Berlin temperatures below -10°C between 1876 - 2019", size=16)
	plt.ylim(-25, -10)
	plt.show()


def days_over_temp():
	data_copy_combined = read_csv_combined()
	# Days with temperatures over a value.
	data_copy_combined['year'] = pd.DatetimeIndex(data_copy_combined['Measurement day']).year
	temp = 25
	over_temp = data_copy_combined[(data_copy_combined['Max daily temp'] > temp)].groupby(['year'])['Measurement day'].count().reset_index()
	print(f"\nDays per year with temperatures over {temp}:\n", over_temp)

	def graph_days_over_temp():
		# Lets show it on a graph
		sns.catplot(
			x='year',
			y='Measurement day',
			data=over_temp,
			kind='bar',
			color='navy'
		)
		plt.title("Berlin: Number of days in a year with temperatures over 25°C.", size=14)
		plt.xlabel("Year", size=11)
		plt.ylabel("Number of days", size=11)
		plt.xticks(rotation=90, size=8)
		plt.yticks(size=8, ticks=(np.arange(0, 89, 2)))
		plt.show()


	def graph_lin_regr_days_over_temp():
		sns.regplot(
			x='year',
			y='Measurement day',
			data=over_temp,
			color='red'
		)
		plt.title("Berlin: Number of days in a year with temperatures over 25°C with lineal regression.", size=14)
		plt.xlabel("Year", size=11)
		plt.ylabel("Number of days", size=11)
		plt.xticks(rotation=90, size=8, ticks=(np.arange(1876, 2021, 2)))
		plt.yticks(size=8, ticks=(np.arange(0, 89, 2)))
		plt.tick_params(labelright=True, size=8) # here improve!!!!
		plt.show()


def convert_to_datetime():
	# Converting the date to datetime if not already is.
	data_copy = select_data()
	data_copy['Measurement day'] = pd.Series.to_numpy(data_copy['Measurement day'])
	print(type(data_copy['Measurement day']))
	print(data_copy.head())
	print("--- Converting to datetime: DONE ---")
	return data_copy

def group_data_decades():
	# Grouping data in decades.
	data_copy_combined = combining_datasets()
	data_copy_combined = data_copy_combined.groupby(pd.cut(data_copy_combined['Measurement day'], pd.date_range('1876', '2019', freq='10YS'), right=False)).mean()
	data_copy_combined.reset_index(inplace=True)
	print("\n--- Grouping data in decades ---")
	print(data_copy_combined)

def select_data_period():
	# Selecting data based on dates; Setting Measurement day as the index.
	data_copy_combined = combining_datasets()
	data_copy_index = data_copy_combined.set_index('Measurement day')
	first_jan = data_copy_index['2019-03-02': '2019-03-02']
	print("\n--- Selected data based on one period ---")
	print(first_jan)
	return data_copy_index

def select_data_many_periods():
	# Selecting many periods to put them together on one plot.
	data_copy_index = select_data_period()
	summer1 = data_copy_index['1970-06-01': '1970-08-31']
	summer2 = data_copy_index['1980-06-01': '1980-08-31']
	summer3 = data_copy_index['1990-06-01': '1990-08-31']
	summer4 = data_copy_index['2000-06-01': '2000-08-31']
	summer5 = data_copy_index['2010-06-01': '2010-08-31']
	summer6 = data_copy_index['2018-06-01': '2018-08-31']

def monthly_data():
	data_copy_index = select_data_period()
	# Aggregating data with resample() and datetime index
	monthly = data_copy_index.resample(rule='M').mean()
	print("\n--- Monthly aggregated mean data ---")
	print(monthly.head())
	monthly.plot(y='Max daily temp', kind='line', lw=0.75, c='r')
	plt.xticks(rotation=45)
	plt.show()

# ## Plot of many plots of periods on one plot.
# # min_temp = 10
# # max_temp = 40
# # lw = 1.5
# # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16,12))
# # ax11 = axes[0][0]
# # ax12 = axes[0][1]
# # ax13 = axes[0][2]
# # ax21 = axes[1][0]
# # ax22 = axes[1][1]
# # ax23 = axes[1][2]
# # summer1.plot(y='Max daily temp', ax=ax11, legend=False, lw=lw, ylim=(min_temp, max_temp))
# # summer2.plot(y='Max daily temp', ax=ax12, legend=False, lw=lw, ylim=(min_temp, max_temp))
# # summer3.plot(y='Max daily temp', ax=ax13, legend=False, lw=lw, ylim=(min_temp, max_temp))
# # summer4.plot(y='Max daily temp', ax=ax21, legend=False, lw=lw, ylim=(min_temp, max_temp))
# # summer5.plot(y='Max daily temp', ax=ax22, legend=False, lw=lw, ylim=(min_temp, max_temp))
# # summer6.plot(y='Max daily temp', ax=ax23, legend=False, lw=lw, ylim=(min_temp, max_temp))
#
# ## Changing the ticks of the axis.
# # yticks = np.arange(start=10, stop=40, step=5)
# # for ax in [ax11, ax12, ax13, ax21, ax22, ax23]:
# # 	# Clear x axis ticks.
# # 	ax.get_xaxis().set_ticks([])
# # 	# Specify y axis ticks.
# # 	ax.yaxis.set_ticks(yticks)
# # 	# Specify major tick label sizes larger.
# # 	ax.tick_params(axis='both', which='major', labelsize=12)
# # for ax in [ax11, ax12, ax13, ax21, ax22, ax23]:
# # 	# Set minor ticks with day numbers.
# # 	ax.xaxis.set_minor_locator(dates.DayLocator(interval=7))
# # 	ax.xaxis.set_minor_formatter(dates.DateFormatter('%d'))
# # 	# Set major ticks with month names.
# # 	ax.xaxis.set_major_locator(dates.MonthLocator())
# # 	ax.xaxis.set_major_formatter(dates.DateFormatter('\n%b'))
#
# ## Adding annotations to each small plot. First defining y und x position.
# # all_y = 12
# # summer1_x = datetime(1970, 8, 15)
# # summer2_x = datetime(1980, 8, 15)
# # summer3_x = datetime(1990, 8, 15)
# # summer4_x = datetime(2000, 8, 15)
# # summer5_x = datetime(2010, 8, 15)
# # summer6_x = datetime(2018, 8, 15)
#
# # ax11.text(summer1_x, all_y, '1970', size=16)
# # ax12.text(summer2_x, all_y, '1980', size=16)
# # ax13.text(summer3_x, all_y, '1990', size=16)
# # ax21.text(summer4_x, all_y, '2000', size=16)
# # ax22.text(summer5_x, all_y, '2010', size=16)
# # ax23.text(summer6_x, all_y, '2018', size=16)
#
# ## Adding a big clear plot behind the small plots to add titel and big one name of y axis.
# # fig.add_subplot(111, frameon=False);
# # plt.grid(False)
# # plt.tick_params(labelcolor='none', length=0, width=0, top='off', bottom='off', left='off', right='off')
# # plt.ylabel("Temperature in Celsius", size=16, family='Arial')
# # plt.title("Summers variations in temperature in Berlin", size=22, family='Arial');
# # plt.tight_layout()
# # plt.savefig("Summer temps in Berlin.png", transparent=True, dpi=600)
# # plt.show()


def all_years_one_plot():
	# Ploting with seaborn
	data_copy_combined = combining_datasets()
	sns.lineplot(x=data_copy_combined['Measurement day'].dt.dayofyear,
						y=data_copy_combined['Max daily temp'],
						hue=data_copy_combined['Measurement day'].dt.year,
						legend='full',
						palette="seismic",
						ci=None,
					);
	plt.title("Berlin maximal daily temperatures 1876 - 2019", size=16)
	plt.legend(ncol=5, loc='lower center', fontsize=9)
	plt.xlim(1, 366)
	plt.ylim(-20, 40)
	plt.show()

def all_years_3d():
	data_3d = read_csv_combined()
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_trisurf(data_3d['Measurement day'].dt.year,
					data_3d['Measurement day'].dt.dayofyear,
					data_3d['Max daily temp'],
					cmap=plt.cm.jet, linewidth=0.2)
	# to Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)
	ax.view_init(30, 45)
	plt.title("Berlin maximal daily temperatures 1876 - 2019, 3d", size=13)
	plt.show()


def calmap_data():
	# Preparation of data for calmap plot.
	data_copy_calmap = combining_datasets()
	data_copy_calmap['year'] = pd.DatetimeIndex(data_copy_calmap['Measurement day']).year
	data_copy_calmap['month'] = pd.DatetimeIndex(data_copy_calmap['Measurement day']).month
	data_copy_calmap['day'] = pd.DatetimeIndex(data_copy_calmap['Measurement day']).day
	print("--- Calmap data: DONE ---")
	return data_copy_calmap

def calmap_calendar_plot():
	# Calmap CALENDAR plot for one defined year.
	data_copy_calmap = calmap_data()
	data_copy_calmap.set_index('Measurement day', inplace=True)
	fig, ax = calmap.calendarplot(data_copy_calmap['2018']['Max daily temp'],
						cmap= 'coolwarm',
						fig_kws={'figsize': (16,10)},
						yearlabel_kws={'color':'black', 'fontsize':24},
						subplot_kws={'title':'Berlin max daily temp'},
						)
	fig.colorbar(ax[0].get_children()[1], ax=ax.ravel().tolist(), orientation='horizontal')
	plt.show()

def calmap_calendar_plot_all_years():
	# Calmap CALENDAR plot for all years (not visible well cause to many years).
	data_copy_calmap = calmap_data()
	data_copy_calmap.set_index('Measurement day', inplace=True)
	fig, ax = calmap.calendarplot(data_copy_calmap['Max daily temp'],
						cmap= 'coolwarm',
						fig_kws={'figsize': (16,10)},
						yearlabel_kws={'color':'black', 'fontsize':24},
						subplot_kws={'title':'Berlin max daily temp'}
						)
	fig.colorbar(ax[0].get_children()[1], ax=ax.ravel().tolist(), orientation='horizontal')
	plt.show()

def calmap_year_plot():
	# Calmap YEAR plot with colorbar.
	data_copy_calmap = calmap_data()
	data_copy_calmap.set_index('Measurement day', inplace=True)
	fig = plt.figure(figsize=(20,8))
	ax = fig.add_subplot(111)
	cax=calmap.yearplot(data_copy_calmap['2018']['Max daily temp'],
						cmap= 'coolwarm',
						ax=ax
						)
	fig.colorbar(cax.get_children()[1], ax=cax, orientation='horizontal')
	plt.show()

def data_go_to_lake():
	# This is a function to prepare data for saying when is warm enough to go to lake.
	data_copy_combined = combining_datasets()
	chosen_temp = 25
	chosen_year = 2019
	data_copy_combined['day_of_week'] = data_copy_combined['Measurement day'].apply(lambda x: x.weekday())
	data_copy_combined['weekend'] = data_copy_combined['day_of_week'] >= 5
	data_copy_combined['hot'] = data_copy_combined['Max daily temp'] >= chosen_temp
	data_copy_combined['go_to_lake_day'] = (data_copy_combined['day_of_week']) & (data_copy_combined['hot'])
	data_copy_combined['go_to_lake_weekend'] = (data_copy_combined['weekend']) & (data_copy_combined['hot'])
	# print(data_copy[150:200])

	# data_copy['year'] = pd.DatetimeIndex(data_copy['Measurement day']).year
	# data_copy['month'] = pd.DatetimeIndex(data_copy['Measurement day']).month
	# data_copy['day'] = pd.DatetimeIndex(data_copy['Measurement day']).day

	lake_any_day = data_copy_combined[data_copy_combined['day_of_week'] >= 0].groupby('Measurement day')['go_to_lake_day'].any()
	lake_any_day = pd.DataFrame(lake_any_day).reset_index()
	lake_weekend = data_copy_combined[data_copy_combined['weekend'] == True].groupby('Measurement day')['go_to_lake_weekend'].any()
	lake_weekend = pd.DataFrame(lake_weekend).reset_index()
	# print(lake_any_day[150:200])

	lake_any_day['year'] = lake_any_day['Measurement day'].apply(lambda x: x.year)
	lake_weekend['year'] = lake_weekend['Measurement day'].apply(lambda x: x.year)
	# lake_weekend['month'] = lake_weekend['Measurement day'].apply(lambda x: x.month)

	yearly_lake_weekend = lake_weekend.groupby('year')['go_to_lake_weekend'].value_counts().rename('days').reset_index()
	yearly_lake_any_day = lake_any_day.groupby('year')['go_to_lake_day'].value_counts().rename('days').reset_index()
	# monthly = lake_weekend.groupby('month')['go_to_lake'].value_counts().rename('days').reset_index()
	# print(yearly_lake_any_day)

	# Filtering only lake weekends or/and other days from one defined year.
	# First possibility is with lambda the second one with pythonic way for-loop.
	# criterion_year = lake_weekend['year'].map(lambda x: x == 2018)
	# lake_weekend_2018=lake_weekend[criterion_year]

	days_hot = lake_any_day[[x==chosen_year for x in lake_any_day['year']] & (lake_any_day['go_to_lake_day'] == True)]
	weekends_hot = lake_weekend[[x==chosen_year for x in lake_weekend['year']] & (lake_weekend['go_to_lake_weekend'] == True)]
	num_days_hot = days_hot.groupby('year')['go_to_lake_day'].value_counts().rename('days').reset_index()
	num_weekends_hot = weekends_hot.groupby('year')['go_to_lake_weekend'].value_counts().rename('days').reset_index()

	lake_weekend_in_year = lake_weekend[[x==chosen_year for x in lake_weekend['year']] & (lake_weekend['go_to_lake_weekend'] == True)]
	# lake_weekend_in_year = lake_weekend[criterion_year & (lake_weekend['go_to_lake'] == True)]

	print(f"In your chosen year {chosen_year} were {num_days_hot['days'].sum()} hot days with more than {chosen_temp}°C.\n"
		f"It means on {num_days_hot['days'].sum()} days you could go to swim in a lake in Berlin.\n"
		f"These days were: \n"
	  	f"{days_hot['Measurement day'].to_string(index=False)}\n"
	  	f"On the other side there were only {num_weekends_hot['days'].sum()} weekend days when you could go to the lakes in {chosen_year}."
	  	f"These weekend days were: \n"
	  	f"{weekends_hot['Measurement day'].to_string(index=False)}")

# # Yearly plot of hot days (lake days)
# sns.set_style("whitegrid")
# sns.barplot(x='year', y='days', hue='go_to_lake_day', data=yearly_lake_any_day)
# plt.xlabel('Year')
# plt.ylabel('Number of days')
# # chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
# plt.show()

# # # Combining into a one excel sheet
# # print("### Getting your data")
# # print("##### Creating an Excel XLSX file. Please wait!")
# # writer = pd.ExcelWriter('ber_rain.xlsx', engine='xlsxwriter')
# # data_copy.to_excel(writer, startrow=1, index=False)
# # writer.save()
# # print("######## The Excel XLSX file has been created.")
