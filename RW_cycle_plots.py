from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import minimize as mini
from scipy.stats import sem
import argparse
import pandas

def pos_load(filename, epoch):
	hdul = fits.open(filename)
	data = hdul[1].data
	
	y_whole = data['PSF_CENTR1']
	x_whole = data['PSF_CENTR2']
	time_whole = data['TIME']
	flags_whole = data['QUALITY']
	
	index = np.where(time_whole > epoch)[0][0]
	time = time_whole[index:]
	x = x_whole[index:]
	y = y_whole[index:]
	flags = flags_whole[index:]
	
	flag_inds = np.where(flags_whole > 0)
	x_flagged = np.delete(x_whole, flag_inds)
	y_flagged = np.delete(y_whole, flag_inds)
	time_flagged = np.delete(time_whole, flag_inds)
	
	return time, x, y, time_flagged, x_flagged, y_flagged

def bg_load(filename, sec):
	hdul = fits.open(filename)
	data = hdul[1].data
	
	time_whole = data['TIME']
	bg_flux_whole = data['SAP_BKG']	
	if sec == 2:	
		index1 = np.where(time_whole > 1369)[0][0]
		index2 = np.where(time_whole > 1377)[0][0]
	elif sec == 1:
		index1 = np.where(time_whole > 1340)[0][0]
		index2 = np.where(time_whole > 1348)[0][0]
		
	time_section = time_whole[index1 : index2]
	bg_section = bg_flux_whole[index1 : index2]
	
	return time_whole, bg_flux_whole, time_section, bg_section

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

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-fn', '--filename', type=str, nargs='*')
	parser.add_argument('-s', '--save', action='store_true')
#	parser.add_argument('-sec', '--sector', type=int)
	parser.add_argument('-bg', '--background', action='store_true')
	parser.add_argument('-c', '--centroid', action='store_true')
	parser.add_argument('-pf', '--phase_fold', action='store_true')
	
	args = parser.parse_args()

	axis_font = {'fontname':'DejaVu Sans', 'size':'15'}
	
	fn = args.filename
	save = args.save
	cen = args.centroid
	bg = args.background
	pf = args.phase_fold
	
	df = pandas.read_csv('/home/astro/phrvdf/tess_data_alerts/TOIs_20181016.csv', index_col='tic_id')		#.csv file containing info on parameters (period, epoch, ID, etc.) of all TOIs
	length = len(df.iloc[0])	

	if cen:
		for i in range(len(fn)):
		
			hdr = fits.open(fn[i])[0].header
			TIC = hdr['TICID']

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

			time, x, y, time_f, x_f, y_f = pos_load(fn[i], epoch)
			phase, phase_order = cycle_fold(time, epoch, 2.5)
		
			fig = plt.figure(figsize=(20, 10))
		
			ax1 = fig.add_subplot(221)
		
			ax1.plot(time_f, x_f, 'bo', markersize=1.5)
			ax1.set_xlabel('Time [BJD - 2457000]', **axis_font)
			ax1.set_ylabel('Row Centroid [pix]', **axis_font)
			ax1.set_title('TIC: {} ; TOI: {} ; Fold Period: 2.5 days'.format(TIC, TOI), **axis_font)
		
			ax2 = fig.add_subplot(222)
		
			ax2.plot(time_f, y_f, 'ro', markersize=1.5)
			ax2.set_xlabel('Time [BJD - 2457000]', **axis_font)
			ax2.set_ylabel('Column Centroid [pix]', **axis_font)
		
			ax3 = fig.add_subplot(223)
		
			ax3.plot(phase, x, 'bo', markersize=1.5)
			ax3.set_xlabel('Phase', **axis_font)
			ax3.set_ylabel('Row Centroid [pix]', **axis_font)
		
			ax4 = fig.add_subplot(224)
		
			ax4.plot(phase, y, 'ro', markersize=1.5)
			ax4.set_xlabel('Phase', **axis_font)
			ax4.set_ylabel('Column Centroid [pix]', **axis_font)
		
			plt.tight_layout()
		
			if save:
				plt.savefig('../Centroid_Positions/Quad_Plots_PSF_positions/TOI_{}_RW_cycle_centroid_quadplot_PSFpositions.png'.format(TOI))
			else:
				plt.show()
			print(TOI)
	
	if bg:
		for i in range(len(fn)):
			
			hdr = fits.open(fn[i])[0].header
			TIC = hdr['TICID']
			cam = hdr['CAMERA']

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

			time, bg_flux, time_section, bg_section = bg_load(fn[i], sec)
			
			if pf:
				phase, phase_order = cycle_fold(time_section, epoch, 2.5)
			
				fig = plt.figure(figsize=(14, 7))
			
				ax1 = fig.add_subplot(211)
			
				ax1.plot(time, bg_flux, 'ko', markersize=1.5)
				ax1.set_xlabel('Time [BJD - 2457000]', **axis_font)
				ax1.set_ylabel('BG Flux [e$^-$ / s]', **axis_font)
				ax1.set_title('TOI: {}  ;   TIC: {}  ; Background Flux; Fold Period: 2.5 days;  Camera: {}'.format(TOI, TIC, cam), **axis_font)
			
				ax2 = fig.add_subplot(212)
			
				ax2.plot(phase, bg_section, 'ko', markersize=1.5)
				ax2.set_xlabel('Phase', **axis_font)
				ax2.set_ylabel('BG Flux [e$^-$ / s]', **axis_font)
			
				plt.tight_layout()
			
				if save:
					plt.savefig('../BG_Plots/TOI_{}_TIC_{}_bgflux.png'.format(TOI, TIC))
				else:
					plt.show()

			else:
				
				plt.figure(figsize=(14, 7))
				
				plt.plot(time, bg_flux, 'ko', markersize=1.5)
				plt.xlabel('Time [BJD - 2457000]', **axis_font)
				plt.ylabel('BG Flux [e$^-$ / s]', **axis_font)
				plt.title('TOI: {}  ;   TIC: {}  ; Background Flux;  Camera:  {}'.format(TOI, TIC, cam), **axis_font)
				
				plt.tight_layout()
			
				if save:
					plt.savefig('../BG_Plots/TOI_{}_TIC_{}_bgflux.png'.format(TOI, TIC))
				else:
					plt.show()

			print(TOI)























