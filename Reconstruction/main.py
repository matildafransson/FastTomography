import glob 
import numpy as np 
import os
import time 
from Nxcreation import NxGenerator

def cutAngleFile(path_angle, flag360):
	"""
	Function to read rotation angle data from a file and categorize it into angle arrays.

	Args:
	    path_angle (str): Path to the angle file.
	    flag360 (bool): Flag indicating whether the angle range is 360 degrees.

	Returns:
	    list: List containing angle arrays and corresponding frame indices.
	"""
	# Read angle data from file
	file = open(path_angle)
	lines = file.readlines()

	# Determine the angle range based on the flag
	if flag360:
		range_angle = 360
	else:
		range_angle = 180

	# Extract the first angle from the file
	first_angle = float(lines[0])

	# Initialize variables
	angle_array_list = []
	angle_array = []
	tomo_n_before = 0 
	list_frameEndBeg = []
	endBeg = [-1,-1]
	flag_new_Tomo = True

	# Iterate through each line in the .txt file
	for i, line in enumerate(lines):

		# Check if a new tomogram starts
		if flag_new_Tomo == True:
			if i == 0 :
				endBeg[0] = i 
			else:
				endBeg[0] = i-1

		# Calculate angle relative to the first angle and the corresponding tomogram number
		angle = (float(line)-first_angle)
		tomo_n_after = angle//range_angle

		# If a new tomogram starts, store the previous angle array and frame indices
		if (tomo_n_after-tomo_n_before):
			endBeg[1] = i-1
			angle_array_list.append(angle_array)
			list_frameEndBeg.append(endBeg)

			# Reset variables for the next tomogram
			endBeg = [-1,-1]
			angle_array = [] 
			flag_new_Tomo = True
		else:
			flag_new_Tomo = False

		# Update tomogram number and calculate angle within the range
		tomo_n_before = tomo_n_after
		angle_u = angle%range_angle
		angle_array.append(angle_u)
		
	return [angle_array_list, list_frameEndBeg]
	

if __name__ == '__main__':

	# Define main folder and other parameters
	main_folder = '/data/projects/whaitiri/Data/JULY_2022/*' #main folder where data is located
	path_dark = main_folder+'TESTS/FT_test1/darks/' #folder where darks are located
	path_ref =  main_folder+'VCT5_FT_N_Exp3'+'/'+'VCT5_FT_N_Exp3_Ref/' #folder where references are located

	flag360 = False
	pixel_size = 40e-6 #Give the pixel size

	energy = 75.0 #Give the X-ray energy (keV)
	distance = 10.0 #Give the detector distance (m)

	list_FT = [] 
	list_folders = glob.glob(main_folder)
	list_folder = []
	# Get list of X-ray folders
	for i, folder in enumerate(list_folders):
		if ('VCT5_FT_N_Exp2' in folder) and not('.txt' in folder): #To select specific experiments to be processed
			list_folder.append(folder)
	list_folder.sort()

	# Iterate through folders
	for i, folder in enumerate(list_folder):
		list_folders_FT_Xray = glob.glob(folder+'/*Xray*/') 
		list_folders_FT_Xray.sort()
		list_folders_FT_Xray = [list_folders_FT_Xray[0]]
		path_angle_file = glob.glob(folder+'/*.txt') 
		path_angle_file.sort()
		nx_file = glob.glob(folder+'/*.nx')

		# Remove existing .nx files
		if len(nx_file) != 0: 
			for file in nx_file: 
				os.remove(file)
							
		if len(list_folders_FT_Xray) == len(path_angle_file) and ((len(list_folders_FT_Xray) != 0) or ((len(path_angle_file) != 0))):
			for i, Xray in enumerate(list_folders_FT_Xray):
				path_prj = Xray
				path_out = folder+'/Scan.nx'
				path_out_angle_array = folder+'/Angle_save.npy'
				path_out_endBeg_array = folder+'/endBeg_save.npy'
				txtFile = path_angle_file[i]

				# Call the cutAngleFile function to process angle data
				array_angles, array_endBeg = cutAngleFile(txtFile,flag360)

				# Calculate total length of angle arrays
				total_length = 0
				for angle in array_angles:
					total_length += len(angle)

				# Initialize NxGenerator and generate .nx files
				generator = NxGenerator(path_prj,path_dark,path_ref,path_out)
				generator.initScanParameter(energy, distance, flag360, pixel_size)
				generator.generateMultiEntryNx(array_angles,array_endBeg)

