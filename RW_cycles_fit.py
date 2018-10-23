import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import minimize as mini
import argparse
import pandas

def fit(X0, time, flux, flux_err):
	a, b, c, d = X0[0], X0[1], X0[2], X0[3]
	model = a + b*time + c*time**2 + d*time**3
	chis = (flux - model)**2 / flux_err**2
	fit_val = np.sum(chis) / (len(chis) - 1)
	
	return fit_val
	
def fit_lin(X0, time, flux, flux_err):
	a, b = X0[0], X0[1]
	model = a + b*time
	chis = (flux - model)**2 / flux_err**2
	fit_val = np.sum(chis) / (len(chis) - 1)
	
	return fit_val

def dataload(filename):
	hdul = fits.open(filename)
	data = hdul[1].data
	
	time_whole = data['TIME']
	gap_index = np.where(np.isnan(time_whole))[0][-1]
	time_main = time_whole[gap_index+1:]
	flux_main = data['SAP_FLUX'][gap_index+1:]
	flux_err_main = data['SAP_FLUX_ERR'][gap_index+1:]
	flags_main = data['QUALITY'][gap_index+1:]

	flag_inds = np.where(flags_main > 0)
	
	time = np.delete(time_main, flag_inds)
	flux = np.delete(flux_main, flag_inds)
	flux_err = np.delete(flux_err_main, flag_inds)
	
	flux1 = np.array([])
	flux2 = np.array([])
	flux3 = np.array([])
	flux4 = np.array([])
	flux5 = np.array([])
	time1 = np.array([])
	time2 = np.array([])
	time3 = np.array([])
	time4 = np.array([])
	time5 = np.array([])
	err1 = np.array([])
	err2 = np.array([])
	err3 = np.array([])
	err4 = np.array([])
	err5 = np.array([])
	
	
	for i in range(len(time)):
		if time[i] <= time[0] + 2.5:
			flux1 = np.append(flux1, flux[i])
			time1 = np.append(time1, time[i])
			err1 = np.append(err1, flux_err[i])
		elif time[i] <= time[0] + 2.5*2:
			flux2 = np.append(flux2, flux[i])
			err2 = np.append(err2, flux_err[i])
			time2 = np.append(time2, time[i])
		elif time[i] <= time[0] + 2.5*3:
			flux3 = np.append(flux3, flux[i])
			err3 = np.append(err3, flux_err[i])
			time3 = np.append(time3, time[i])
		elif time[i] <= time[0] + 2.5*4:
			flux4 = np.append(flux4, flux[i])
			err4 = np.append(err4, flux_err[i])
			time4 = np.append(time4, time[i])
		elif time[i] <= time[0] + 2.5*5:
			flux5 = np.append(flux5, flux[i])
			time5 = np.append(time5, time[i])
			err5 = np.append(err5, flux_err[i])
	
	time1 -= time1[0]
	time2 -= time2[0]
	time3 -= time3[0]
	time4 -= time4[0]
	time5 -= time5[0]
	
	return time1, flux1, err1, time2, flux2, err2, time3, flux3, err3, time4, flux4, err4, time5, flux5, err5
		
def data_bin(time, flux, no_bins):
	data_array = np.zeros((len(time), 3))
	data_array[:, 0], data_array[:, 1] = time, flux
	for i in range(len(time)):
		data_array[i, 2] = np.int(time[i] * no_bins / time[-1])
	
	time_binned = np.zeros(no_bins)
	flux_binned = np.zeros(no_bins)
	
	for j in range(no_bins):
		time_binned[j] = np.mean(data_array[np.where(data_array[:, 2] == i), 0])
		flux_binned[j] = np.mean(data_array[np.where(data_array[:, 2] == i), 1])
	
	return time_binned, flux_binned

def poly_fit(time, flux, flux_err):
	
	a0 = flux[0]
	b0 = -10
	c0 = 4
	d0 = 5
	
	X0 = [a0, b0, c0, d0]
	
	result = mini(fit, X0, args=(time, flux, flux_err))
	
	for i in range(10):
		result = mini(fit, result.x, args=(time, flux, flux_err))
#		print result.x
	
	poly_model = result.x[0] + result.x[1]*time + result.x[2]*time**2 + result.x[3]*time**3
	
	return poly_model, result.x, result.fun
	
def lin_fit(time, flux, flux_err):
	
	a0 = flux[0]
	b0 = (flux[-1] - flux[0]) / (time[-1] - time[0])
	
	X0 = [a0, b0]
	
	result = mini(fit_lin, X0, args=(time, flux, flux_err))
	
	for i in range(10):
		result = mini(fit_lin, result.x, args=(time, flux, flux_err))
		
	lin_model = result.x[0] + result.x[1]*time
	
	return lin_model, result.x, result.fun
	
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-fn', '--filename', type=str, nargs='*')
	parser.add_argument('-s', '--save', action='store_true')
	parser.add_argument('-l', '--linear', action='store_true')
	
	args = parser.parse_args()
	
	fn = args.filename
	save = args.save
	lin = args.linear

	axis_font = {'fontname':'DejaVu Sans', 'size':'15'}
	
	fn = args.filename

	df = pandas.read_csv('/home/astro/phrvdf/tess_data_alerts/TOIs_20181016.csv', index_col='tic_id')		#.csv file containing info on parameters (period, epoch, ID, etc.) of all TOIs
	length = len(df.iloc[0])	
	

	for i in range(len(fn)):
		
		hdr = fits.open(fn[i])[0].header
		TIC = hdr['TICID']
		
		df2 = df.loc[TIC]
		if len(df2) == length:
			TOI = np.int(df2.loc['toi_id'])            	  	#TIC ID for the object - used for plot title
		else:
			df3 = df2.iloc[0]
			TOI = np.int(df3.loc['toi_id'])

		time1, flux1, err1, time2, flux2, err2, time3, flux3, err3, time4, flux4, err4, time5, flux5, err5 = dataload(fn[i])
		
		if lin:
			
			lin_model1, coeffs1, chi1 = lin_fit(time1, flux1, err1)
			lin_model2, coeffs2, chi2 = lin_fit(time2, flux2, err2)
			lin_model3, coeffs3, chi3 = lin_fit(time3, flux3, err3)
			lin_model4, coeffs4, chi4 = lin_fit(time4, flux4, err4)
			lin_model5, coeffs5, chi5 = lin_fit(time5, flux5, err5)
			
			fig = plt.figure(figsize=(20, 10))
			
			ax1 = fig.add_subplot(321)
			
			ax1.plot(time1, flux1, 'ko', markersize=1)
			ax1.plot(time1, lin_model1, 'r--')
			ax1.plot(time2 + 2.5, flux2, 'ko', markersize=1)
			ax1.plot(time2 + 2.5, lin_model2, 'r--')
			ax1.plot(time3 + 5.0, flux3, 'ko', markersize=1)
			ax1.plot(time3 + 5.0, lin_model3, 'r--')
			ax1.plot(time4 + 7.5, flux4, 'ko', markersize=1)
			ax1.plot(time4 + 7.5, lin_model4, 'r--')
			ax1.plot(time5 + 10.0, flux5, 'ko', markersize=1)
			ax1.plot(time5 + 10.0, lin_model5, 'r--')
		
			ax1.set_xlabel('Time [days]', **axis_font)
			ax1.set_ylabel('SAP Flux [e$^-$ / s]', **axis_font)
			ax1.set_title('TIC: {} ;  TOI: {}'.format(TIC, TOI), **axis_font)
		
			ax2 = fig.add_subplot(322)
			ax2.plot(time1, flux1, 'ko', markersize=1)
			ax2.plot(time1, lin_model1, 'r--')
			ax2.set_xlabel('Time [days]', **axis_font)
			ax2.set_ylabel('SAP Flux [e$^-$ / s]', **axis_font)
			ax2.set_title('RW Cycle 1. (m = {:.2f} ;  c = {:.2f} ;  $\chi^2$ = {:.2f})'.format(coeffs1[1], coeffs1[0], chi1), **axis_font)
		
			ax3 = fig.add_subplot(323)
			ax3.plot(time2, flux2, 'ko', markersize=1)
			ax3.plot(time2, lin_model2, 'r--')
			ax3.set_xlabel('Time [days]', **axis_font)
			ax3.set_ylabel('SAP Flux [e$^-$ / s]', **axis_font)
			ax3.set_title('RW Cycle 2. (m = {:.2f} ;  c = {:.2f} ;  $\chi^2$ = {:.2f})'.format(coeffs2[1], coeffs2[0], chi2), **axis_font)
			
			ax4 = fig.add_subplot(324)
			ax4.plot(time3, flux3, 'ko', markersize=1)
			ax4.plot(time3, lin_model3, 'r--')
			ax4.set_xlabel('Time [days]', **axis_font)
			ax4.set_ylabel('SAP Flux [e$^-$ / s]', **axis_font)
			ax4.set_title('RW Cycle 3. (m = {:.2f} ;  c = {:.2f} ;  $\chi^2$ = {:.2f})'.format(coeffs3[1], coeffs3[0], chi3), **axis_font)
			
			ax5 = fig.add_subplot(325)
			ax5.plot(time4, flux4, 'ko', markersize=1)
			ax5.plot(time4, lin_model4, 'r--')
			ax5.set_xlabel('Time [days]', **axis_font)
			ax5.set_ylabel('SAP Flux [e$^-$ / s]', **axis_font)
			ax5.set_title('RW Cycle 4. (m = {:.2f} ;  c = {:.2f} ;  $\chi^2$ = {:.2f})'.format(coeffs4[1], coeffs4[0], chi4), **axis_font)
		
			ax6 = fig.add_subplot(326)
			ax6.plot(time5, flux5, 'ko', markersize=1)
			ax6.plot(time5, lin_model5, 'r--')
			ax6.set_xlabel('Time [days]', **axis_font)
			ax6.set_ylabel('SAP Flux [e$^-$ / s]', **axis_font)
			ax6.set_title('RW Cycle 5. (m = {:.2f} ;  c = {:.2f} ;  $\chi^2$ = {:.2f})'.format(coeffs5[1], coeffs5[0], chi5), **axis_font)
			
			plt.tight_layout()
		
			if save:
				plt.savefig('../RW_cycles_linearfits/TOI_{}_TIC_{}_cycles_linfit.png'.format(TOI, TIC))
			else:
				plt.show()
			
			print TOI
			
			
		else:	
		
			poly_model1, coeffs1, chi1 = poly_fit(time1, flux1, err1)
			poly_model2, coeffs2, chi2 = poly_fit(time2, flux2, err2)
			poly_model3, coeffs3, chi3 = poly_fit(time3, flux3, err3)
			poly_model4, coeffs4, chi4 = poly_fit(time4, flux4, err4)
			poly_model5, coeffs5, chi5 = poly_fit(time5, flux5, err5)
		
			fig = plt.figure(figsize=(20, 10))
		
			ax1 = fig.add_subplot(321)
		
			ax1.plot(time1, flux1, 'ko', markersize=1)
			ax1.plot(time1, poly_model1, 'r--')
			ax1.plot(time2 + 2.5, flux2, 'ko', markersize=1)
			ax1.plot(time2 + 2.5, poly_model2, 'r--')
			ax1.plot(time3 + 5.0, flux3, 'ko', markersize=1)
			ax1.plot(time3 + 5.0, poly_model3, 'r--')
			ax1.plot(time4 + 7.5, flux4, 'ko', markersize=1)
			ax1.plot(time4 + 7.5, poly_model4, 'r--')
			ax1.plot(time5 + 10.0, flux5, 'ko', markersize=1)
			ax1.plot(time5 + 10.0, poly_model5, 'r--')
		
			ax1.set_xlabel('Time [days]', **axis_font)
			ax1.set_ylabel('SAP Flux [e$^-$ / s]', **axis_font)
			ax1.set_title('TIC: {} ;  TOI: {}'.format(TIC, TOI), **axis_font)
		
			ax2 = fig.add_subplot(322)
			ax2.plot(time1, flux1, 'ko', markersize=1)
			ax2.plot(time1, poly_model1, 'r--')
			ax2.set_xlabel('Time [days]', **axis_font)
			ax2.set_ylabel('SAP Flux [e$^-$ / s]', **axis_font)
			ax2.set_title('RW Cycle 1. (a={:.2f}, b={:.2f}, c={:.2f}, d={:.2f}, $\chi^2$={:.2f})'.format(coeffs1[0], coeffs1[1], coeffs1[2], coeffs1[3], chi1), **axis_font)
		
			ax3 = fig.add_subplot(323)
			ax3.plot(time2, flux2, 'ko', markersize=1)
			ax3.plot(time2, poly_model2, 'r--')
			ax3.set_xlabel('Time [days]', **axis_font)
			ax3.set_ylabel('SAP Flux [e$^-$ / s]', **axis_font)
			ax3.set_title('RW Cycle 2. (a={:.2f}, b={:.2f}, c={:.2f}, d={:.2f}, $\chi^2$={:.2f})'.format(coeffs2[0], coeffs2[1], coeffs2[2], coeffs2[3], chi2), **axis_font)
			
			ax4 = fig.add_subplot(324)
			ax4.plot(time3, flux3, 'ko', markersize=1)
			ax4.plot(time3, poly_model3, 'r--')
			ax4.set_xlabel('Time [days]', **axis_font)
			ax4.set_ylabel('SAP Flux [e$^-$ / s]', **axis_font)
			ax4.set_title('RW Cycle 3. (a={:.2f}, b={:.2f}, c={:.2f}, d={:.2f}, $\chi^2$={:.2f})'.format(coeffs3[0], coeffs3[1], coeffs3[2], coeffs3[3], chi3), **axis_font)
			
			ax5 = fig.add_subplot(325)
			ax5.plot(time4, flux4, 'ko', markersize=1)
			ax5.plot(time4, poly_model4, 'r--')
			ax5.set_xlabel('Time [days]', **axis_font)
			ax5.set_ylabel('SAP Flux [e$^-$ / s]', **axis_font)
			ax5.set_title('RW Cycle 4. (a={:.2f}, b={:.2f}, c={:.2f}, d={:.2f}, $\chi^2$={:.2f})'.format(coeffs4[0], coeffs4[1], coeffs4[2], coeffs4[3], chi4), **axis_font)
		
			ax6 = fig.add_subplot(326)
			ax6.plot(time5, flux5, 'ko', markersize=1)
			ax6.plot(time5, poly_model5, 'r--')
			ax6.set_xlabel('Time [days]', **axis_font)
			ax6.set_ylabel('SAP Flux [e$^-$ / s]', **axis_font)
			ax6.set_title('RW Cycle 5. (a={:.2f}, b={:.2f}, c={:.2f}, d={:.2f}, $\chi^2$={:.2f})'.format(coeffs5[0], coeffs5[1], coeffs5[2], coeffs5[3], chi5), **axis_font)
			
			plt.tight_layout()
		
			if save:
				plt.savefig('../RW_cycles_polyfits/TOI_{}_TIC_{}_cycles_polyfit.png'.format(TOI, TIC))
			else:
				plt.show()
			
			print TOI
		

		
		
		


















