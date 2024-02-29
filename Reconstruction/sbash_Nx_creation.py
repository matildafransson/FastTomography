import tomoscan.nexus
from nxtomomill.nexus.nxtomo import NXtomo
from nxtomomill.utils import ImageKey
from tomoscan.esrf import HDF5TomoScan
from tomoscan.validator import ReconstructionValidator
import sys, getopt
from PIL import Image 
import numpy as np
import glob
import os


class NxGenerator():
	def __init__(self,path_prj,path_dark,path_ref,path_out):

		self.path_prj = path_prj
		self.path_dark = path_dark
		self.path_ref = path_ref
		self.path_out = path_out
		
		if os.path.exists(self.path_out):
		    os.remove(self.path_out)


		self.list_prj = glob.glob(self.path_prj + '*.tif' )

		self.list_dark = glob.glob(self.path_dark + '*.tif')

		self.list_ref = glob.glob(self.path_ref + '*.tif')

		self.list_dark.sort()
		self.list_ref.sort()
		self.list_prj.sort()


		assert len(self.list_prj) != 0
		assert len(self.list_dark) != 0
		assert len(self.list_ref) != 0

		self.nb_dark_total = len(self.list_dark)
		self.nb_ref_total = len(self.list_ref)
		self.nb_prj_total = len(self.list_prj)


	def initScanParameter(self, energy, distance, flag360, pixel_size):
		self.energy = energy
		self.distance = distance
		self.pixel_size = pixel_size
		self.flag_360 = flag360


	def generateSingleNXEntry(self, indexTomo, angle_array, endBeg):


		index_dark = [0,self.nb_dark_total-1]
		index_ref  = [0,self.nb_ref_total-1]
		
		self.nbFrameTomo = len(angle_array)
		self.angle_array = angle_array

		start_prj = endBeg[0]
		end_prj =  endBeg[1]

		assert end_prj < self.nb_prj_total


		index_prj  = [start_prj,end_prj]

		nb_dark = index_dark[1] - index_dark[0] + 1
		nb_ref  = index_ref[1]  - index_ref[0] + 1


		im = np.array(Image.open(self.list_prj[0]))
		im_shape = im.shape

		array_dark = np.zeros((nb_dark, im_shape[0],im_shape[1]))
		array_ref =  np.zeros((nb_ref, im_shape[0],im_shape[1]))
		array_prj =  np.zeros((self.nbFrameTomo, im_shape[0],im_shape[1]))

		for i in range(index_dark[0],index_dark[1]+1):
			print('Loading Dark Image ', str(i))
			array_dark[i-index_dark[0]] = Image.open(self.list_dark[i])

		for i in range(index_ref[0],index_ref[1]+1):
			print('Loading Ref Image ', str(i))
			array_ref[i-index_ref[0]] = Image.open(self.list_ref[i])

		for i in range(index_prj[0],index_prj[1]+1):
			print('Loading Prj Image ', str(i))
			array_prj[i-index_prj[0]] = Image.open(self.list_prj[i])
	
		self.entryName = 'entry'+str(indexTomo).zfill(4)
		my_nxtomo = NXtomo('')
		
		if self.flag_360:
			my_nxtomo.instrument.detector.field_of_view = "Half"
		else:
			my_nxtomo.instrument.detector.field_of_view = "Full"



		data = np.concatenate([array_dark,array_ref,array_prj])
		my_nxtomo.instrument.detector.data = data

		image_key_control = np.concatenate([
		[ImageKey.DARK_FIELD]*nb_dark,
		[ImageKey.FLAT_FIELD]*nb_ref,
		[ImageKey.PROJECTION]*self.nbFrameTomo,
		])

		assert len(image_key_control) == len(data)
		my_nxtomo.instrument.detector.image_key_control = image_key_control

		rotation_angle = np.concatenate([
			[0.0]*nb_dark,
			[0.0]*nb_ref,
			self.angle_array,
		])


		assert len(rotation_angle) == len(data)

		my_nxtomo.sample.rotation_angle = rotation_angle

		my_nxtomo.instrument.detector.x_pixel_size = my_nxtomo.instrument.detector.y_pixel_size = self.pixel_size

		my_nxtomo.energy = self.energy
		my_nxtomo.instrument.detector.distance = self.distance


		my_nxtomo.save(file_path=self.path_out, data_path=self.entryName, overwrite=False)

	def H5Validation(self):
		scan = HDF5TomoScan(self.path_out, entry=self.entryName)
		validator = ReconstructionValidator(scan, check_phase_retrieval=False, check_values=True)
		assert validator.is_valid()

	def generateMultiEntryNx(self,array_angles,array_endBeg):
		
		for i, angles in enumerate(array_angles):
			self.generateSingleNXEntry(i,angles,array_endBeg[i])
			

		

if __name__ == '__main__':

    opts, args = None, None

    try:
        opts, args = getopt.getopt(sys.argv[2:], 'p:b:r:o:e:d:fx:a:c:m:t:',
                                   ["path_prj=", "path_dark=", "path_ref=", "path_out=",
                                    "energy=", 'distance=', 'flag360', 'pixel_size=', "array_angles=", 'array_endBeg=', 'rec_folder=','cor='])
    except getopt.GetoptError:
        print("Unknown argument")
        
    print(opts,sys.argv)


    path_prj = 'None'
    path_dark = 'None'
    path_ref = 'None'
    path_out= 'None'
    energy = 'None'
    distance = 'None'
    flag360 = False
    pixel_size = 'None'
    array_angles = 'None'
    array_endBeg = 'None' 
    folder = 'None'
    cor = 'None'

    for option, value in opts:

        if option in ('-p', '--path_prj'):
            path_prj = value+'/*.tif' 
            
        if option in ('-b', '--path_dark'):
            path_dark = value+'/*.tif' 
            
        if option in ('-r', '--path_ref'):
            path_ref = value+'/*.tif'     
              
        if option in ('-o', '--path_out'):
            path_out = value
            
        if option in ('-e', '--energy'):
            energy = float(value) 
            
        if option in ('-d', 'distance'):
            distance = float(value)   
            
        if option in ('-f', '--flag360'):
            flag360 = True
            
        if option in ('-x', '--pixel_size'):
            pixel_size = float(value) 
            
        if option in ('-a', '--array_angles'):
            array_angles = value  
            
        if option in ('-c', '--array_endBeg'):
            array_endBeg = value    
            
        if option in ('-m', '--rec_folder'):
            folder = value
            
        if option in ('-t', '--cor'):
            cor = value


    angles = np.load(array_angles,allow_pickle = True)
    endBeg = np.load(array_endBeg, allow_pickle = True)
    
    generator = NxGenerator(path_prj,path_dark,path_ref,path_out)
    generator.initScanParameter(energy, distance, flag360, pixel_size)
    generator.generateMultiEntryNx(angles,endBeg)
    
    #out_path = '/data/projects/whaitiri/Data/JULY_2022/ReconFastTomo/' +os.path.basename(folder) +'/' 
    #run_recon(folder, out_path,cor)
    


