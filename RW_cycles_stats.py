from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import minimize as mini
from scipy.stats import sem
import argparse
import pandas

def chi_squared(X0, x, y, yerr):
	m = X0[0]
	c = X0[1]
	mod = m*x + c
	chis = (y - mod)**2 / (yerr**2)
	chi2 = np.sum(chis) / (len(chis - 1))
	
	return chi2

def dataload(filename):
	hdul = fits.open(filename)
	data = hdul[1].data
	
	time_whole = data['TIME']
	gap_index = np.where(np.isnan(time_whole))[0][-1]
#	gap_index = np.where(time_whole > 1371.13)[0][0]
	time_main = time_whole[gap_index:]
	flux_main = data['SAP_FLUX'][gap_index:]
	flags_main = data['QUALITY'][gap_index:]

	flag_inds = np.where(flags_main > 0)
	
	time = np.delete(time_main, flag_inds)
	flux = np.delete(flux_main, flag_inds)
	
	return time, flux
	
def pos_lod(filename):
	hdul = fits.open(filename)
	data = hdul[1].data
	
	y = data['MOM_CENTR1']
	x = data['MOM_CENTR2']
	time = data['TIME']
	
	null_indices = np.where(np.isnan(time))
	time = np.delete(time, null_indices)
	x = np.delete(x, null_indices)
	y = np.delete(y, null_indices)
	
	return time, x, y
	

def cycle_fold(time, epoch, period=2.5):
	phase = np.zeros_like(time)
	phase_order = np.zeros_like(time)
	for i in range(len(phase)):
		phase[i] = (time[i] - epoch) / period  -  np.int((time[i] - epoch) / period)
		phase_order[i] = (time[i] - epoch) / period  -  np.int((time[i] - epoch) / period)
	
	for i in range(len(phase)):
		
		if phase[i] < -0.25:
			phase[i] += 1.
			
		elif phase[i] > 0.75:
			phase[i] -= 1.
	for j in range(len(phase_order)):
		if phase_order[j] < 0:
			phase_order[j] += 1.
	
	return phase, phase_order

def cycle_stats(phase, flux, no_bins = 10):
	stats_array = np.zeros((len(phase), 3))
	stats_array[:, 0], stats_array[:, 1] = phase, flux
	for i in range(len(flux)):
		stats_array[i, 2] = np.int(phase[i] * no_bins)
	
	phase_means = np.zeros(no_bins)
	flux_means = np.zeros(no_bins)
	flux_sigs = np.zeros(no_bins)
	flux_sem = np.zeros(no_bins)
	
	for j in range(no_bins):
		bin_vals = np.where(stats_array[:, 2] == j)
		phase_means[j] = np.mean(stats_array[bin_vals, 0])
		flux_means[j] = np.mean(stats_array[bin_vals, 1])
		flux_sigs[j] = np.std(stats_array[bin_vals, 1])
		flux_sem[j] = sem(stats_array[bin_vals[0], 1])
		
	return phase_means, flux_means, flux_sigs, flux_sem
	

def best_params_find(X0, x, y, yerr):
	result = mini(chi_squared, X0, args=(x, y, yerr))
	for i in range(10):
		result = mini(chi_squared, result.x, args=(x, y, yerr))
		
	return result.x, result.fun


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-fn', '--filename', type=str, nargs='*')
#	parser.add_argument('-sec', '--sector', type=int)
	
	args = parser.parse_args()

	axis_font = {'fontname':'DejaVu Sans', 'size':'15'}
	
	fn = args.filename

	df = pandas.read_csv('/home/astro/phrvdf/tess_data_alerts/TOIs_20181016.csv', index_col='tic_id')		#.csv file containing info on parameters (period, epoch, ID, etc.) of all TOIs
	length = len(df.iloc[0])	
	
	for i in range(len(fn)):
		
		hdr = fits.open(fn[i])[0].header
		
		if hdr['ORIGIN'] == 'NASA/Ames':
			TIC = hdr['TICID']
		elif hdr['ORIGIN'] == 'MIT/QLP':
			TIC = np.int(hdr['TIC'])
		
		camera = hdr['CAMERA']
		
		sec = np.int(hdr['SECTOR'])
		if sec == 2:
			epoch = 1371.13
		elif sec == 1:
			epoch = 1342.19
			
		df2 = df.loc[TIC]
		if len(df2) == length:
			TOI = np.int(df2.loc['toi_id'])            	  	#TIC ID for the object - used for plot title
		else:
			df3 = df2.iloc[0]
			TOI = np.int(df3.loc['toi_id'])
		
		time, flux = dataload(fn[i])
		phase, phase_order = cycle_fold(time, epoch)
		pmeans, fmeans, fsigs, fsem = cycle_stats(phase_order, flux)
		
		m0 = (fmeans[-1] - fmeans[0]) / (pmeans[-1] - pmeans[0])
		c0 = fmeans[0] - pmeans[0] * m0
		
		X0 = [m0, c0]
		
		params_best, chi = best_params_find(X0, pmeans, fmeans, fsem)
		mod = pmeans * params_best[0] + params_best[1]
		p_set = np.linspace(0, 0.75, 250)
		mod_set = params_best[0] * p_set + params_best[1]
		print params_best, chi
		
		fig = plt.figure(figsize=(14, 7))
		
		ax1 = fig.add_subplot(121)
		ax1.plot(phase, flux, 'ko', markersize=1.5)
		ax1.plot(p_set, mod_set, 'r--')
		ax1.set_xlabel('Phase', **axis_font)
		ax1.set_ylabel('SAP Flux [e$^-$ / s]', **axis_font)
		ax1.set_title('TIC : {}  ;  Fold Period : 2.5 days ;  TESS Camera no.: {}'.format(TIC, camera), **axis_font)

		ax2 = fig.add_subplot(122)
		ax2.errorbar(pmeans, fmeans, yerr=fsem, marker='o', color='black', linestyle='none')
		ax2.plot(pmeans, mod, 'r--')
		ax2.set_xlabel('Phase', **axis_font)
		ax2.set_ylabel('SAP Flux [e$^-$ / s]', **axis_font)
		
		ax2.set_title('m = {:.3f} ;  c = {:.3f} ;  $\chi^2$ : {:.3f}'.format(params_best[0], params_best[1], chi), **axis_font)
		plt.savefig('RW_cycle_plots2/TOI_{}_RWcycles.png'.format(TOI))
		






















