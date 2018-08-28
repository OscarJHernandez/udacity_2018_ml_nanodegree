import numpy as np
import matplotlib.pyplot as plt
import csv
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.tools.plotting import autocorrelation_plot
import pandas as pd

# First we read in the sunspot data
fileName = "data/wolfer-sunspot-numbers-1770-to-1.csv"

years = []
number_of_sunspots = []


with open(fileName, newline='') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
	
	for row in spamreader:
		
		try:
			print(int(row[0]),float(row[1]))
			years.append(str(int(row[0]))+"-01-02")
			number_of_sunspots.append(float(row[1]))
		except:
			pass


# Now we create a pandas object
df = pd.DataFrame({"Year": years,"number of sunspots": number_of_sunspots})
df['Year'] = pd.to_datetime(df['Year'])
df  = df.set_index("Year")

print(df.head())
print(pd.infer_freq(df.index))

result = seasonal_decompose(df, model='additive',freq=1)
result.plot()
plt.show()

plt.clf()
autocorrelation_plot(df)
plt.show()
