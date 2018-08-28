#----------------------------------------------------------------------------
#
# Author: Oscar Javier Hernandez
#
#----------------------------------------------------------------------------
# This subroutine will fit ARIMA, and Seasonal ARIMA models to data
import itertools
import statsmodels.api as sm
import statsmodels
from pandas.tools.plotting import autocorrelation_plot
import numpy as np


def rmse(predictions, targets):
	'''
	The root-mean squared value
	'''
	return np.sqrt(((predictions - targets) ** 2).mean())

	
def fit_SARIMAX(pMax,dMax,qMax,t,train_data,test_data,minimize="RMS"):
	'''
	This function will carry out a gridsearch and fit a seasonal SARIMA model based on RMS values
	'''
	
	AIC = []
	RMS = []
	SARIMAX_model = []
	
	p = range(0,pMax+1)
	d = range(0,dMax+1)
	q = range(0,qMax+1)
	t = [t]
	
	# This creates all combinations of p,q,d
	pdq = list(itertools.product(p, d, q))
	
	# This creates all combinations of the seasonal variables
	seasonal_pdq = [(x[0], x[1], x[2], x[3]) for x in list(itertools.product(p, d, q,t))]
	
	# Make an array of the true data
	actual = np.asarray([test_data.iloc[:,0][i] for i in range(len(test_data))])
	
	# Now we iterate over all parameters to find a good fit 
	for param in pdq:
		for param_seasonal in seasonal_pdq:
			
			try:
				mod = sm.tsa.statespace.SARIMAX(train_data,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

				results = mod.fit()
				
				# Compute the R2 value
				predictions = results.get_forecast(test_data.index[-1]).predicted_mean 
				
				# Convert the predictions to an array
				predictions = np.asarray([predictions[i] for i in range(len(predictions))])
				
				     
				rms = rmse(actual, predictions)

				print('SARIMAX{}x{} - AIC:{} - RMS:{}'.format(param, param_seasonal, results.aic,rms), end='\r')
				
				
				AIC.append(results.aic)
				SARIMAX_model.append([param, param_seasonal])
				RMS.append(rms)
			except:
				continue
	
	if(minimize=='RMS'):
		min_value = min(RMS)
		min_index = RMS.index(min(RMS))
		print('The smallest RMS is {} for model SARIMAX{}x{}'.format(min_value, SARIMAX_model[min_index][0],SARIMAX_model[min_index][1]))
	elif(minimize=='AIC'):
		min_value = min(AIC)
		min_index = AIC.index(min(AIC))
		print('The smallest AIC is {} for model SARIMAX{}x{}'.format(min_value, SARIMAX_model[min_index][0],SARIMAX_model[min_index][1]))
		
	
	
	# The optimal model is then fitted
	mod = sm.tsa.statespace.SARIMAX(train_data,
	                                order=SARIMAX_model[min_index][0],
	                                seasonal_order=SARIMAX_model[min_index][1],
	                                enforce_stationarity=False,
	                                enforce_invertibility=False)
	
	results = mod.fit()
	
	
	return AIC,RMS,SARIMAX_model,results
	
	
	
def fit_ARIMAX(pMax,dMax,qMax,train_data,test_data,minimize="RMS"):
	'''
	This function fits an ARIMA model to data
	'''
	ARMA_model = []
	AIC = []
	RMS = []
	
	p = range(0,pMax+1)
	d = range(0,dMax+1)
	q = range(0,qMax+1)
	
	# This creates all combinations of p,q,d
	pdq = list(itertools.product(p, d, q))
	
	actual = np.asarray([test_data.iloc[:,0][i] for i in range(len(test_data))])

	for param in pdq:
	    try:
	        arma_mod = statsmodels.tsa.arima_model.ARIMA(train_data,order=param).fit()
	        aic = arma_mod.aic
	        
	        AIC.append(aic)
	        ARMA_model.append([param])
	        
	        # Compute the R2 value
	        predictions = arma_mod.predict(start=test_data.index[0], end=test_data.index[-1])       
	        rms = rmse(actual, predictions)
	        
	        RMS.append(rms)
	        
	        print(param,aic,rms,end='\r')
	    
	    except:
	        continue
	
	
	if(minimize=='RMS'):
		min_value = min(RMS)
		min_index = RMS.index(min(RMS))
		print('The smallest RMS is {} for model ARMA{}'.format(min_value, ARMA_model[min_index][0]))
	elif(minimize=='AIC'):
		min_value = min(AIC)
		min_index = AIC.index(min(AIC))
		print('The smallest AIC is {} for model ARMA{}'.format(min_value, ARMA_model[min_index][0]))

	
	results = statsmodels.tsa.arima_model.ARIMA(train_data, ARMA_model[min_index][0]).fit(disp=False)

	
	return AIC,RMS,ARMA_model,results
