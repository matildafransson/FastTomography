import numpy as np
from PIL import Image
import os
import scipy.ndimage
from scipy.ndimage import median_filter
import fabio.tifimage as tif
import SimpleITK as sitk

class Volume_Rotator():
    def __init__(self, image, center, threshold, seg_min_value, seg_max_value, seedListToSegment,size_median_filter):

        self.image = image
        self.center = center
        self.threshold = threshold
        self.seg_min_value = seg_min_value
        self.seg_max_value = seg_max_value
        self.seedListToSegment = seedListToSegment
        self.size_median_filter  = size_median_filter

    def find_rotation(self):
        # Find center of image and shift this to the center
        image = Image.open(self.image)
        print('Opened image')
        img = np.array(image)
        img = np.interp(img, (img.min(), img.max()), (1, 0))
        img2 = self.shift_array_to_center(img, center)

        #Segment only the battery
        segmented_image = self.image_segmentation(img2, seg_min_value, seg_max_value, seedListToSegment)
        mask = self.create_mask(segmented_image, max_r+5)
        img2 = segmented_image * mask

        #Apply median filter and create new mask
        img2 = median_filter(img2, size_median_filter)
        mask = self.create_mask(img2, max_r)
        img2 = img2 * mask

        img2[img2<=threshold] = 0
        img2[img2 > threshold] = 1

        labels = scipy.ndimage.label(img2)

        max_size_label = -1
        label_max = -1

        if labels[1] != 1:
            for label in range(1,labels[1]):
                size = np.sum(labels[0] == label)
                if size > max_size_label:
                    max_size_label = size
                    label_max = label

            img2 = labels[0]
            img2[img2 != label_max] = 0

        cm = self.find_center_of_mass(img2)
        alpha = self.find_angle(center, cm)

        return alpha, center

    def shift_array_to_center(self,array, point_position):
        rows, cols = array.shape
        center_x = rows // 2
        center_y = cols // 2
        current_x, current_y = point_position

        displacement_x = center_x - current_x
        displacement_y = center_y - current_y

        shifted_array = np.zeros_like(array)

        for i in range(rows):
            for j in range(cols):
                shifted_i = i + displacement_x
                shifted_j = j + displacement_y
                if 0 <= shifted_i < rows and 0 <= shifted_j < cols:
                    shifted_array[shifted_i, shifted_j] = array[i, j]

        return shifted_array


    def image_segmentation(self, image, val_min, val_max, seedListToSegment):
        segmented_1 = self.SegConnectedThreshold(image, val_min, val_max, seedListToSegment)
        im_1 = np.multiply(segmented_1, image)
        return im_1


    def create_mask(self, img, max_r):
        '''
        Args:
            img: Image to mask
            max_r: Distance from the center of the battery to the

        Returns: Masked image
        '''
        shape = img.shape
        mask = np.zeros_like(img)

        # Define the center coordinates of the circle
        center_x, center_y = (img.shape[0] // 2, img.shape[0] // 2)

        # Generate the circle by setting the appropriate array elements to 1
        for i in range(shape[0]):
            for j in range(shape[1]):
                distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                if distance <= max_r:
                    mask[i, j] = True
                else:
                    mask[i, j] = False

        return mask

    def SegConnectedThreshold(self, vol, val_min, val_max, seedListToSegment):
        '''
        Args:
            vol: volume to be segmented
            val_min: minimum greyscale value (threshold)
            val_max: maximum greyscale value (threshold)
            seedListToSegment: initial list of [x,y,z] coordinates as starting point for the segmentation

        Returns: Segmented volume
        '''
        ITK_Vol = self.imageFromNumpyToITK(vol)
        segmentationFilter = sitk.ConnectedThresholdImageFilter()

        for seed in seedListToSegment:
            seedItk = (seed[0], seed[1], seed[2])
            segmentationFilter.AddSeed(seedItk)

        segmentationFilter.SetLower(val_min)
        segmentationFilter.SetUpper(val_max)
        segmentationFilter.SetReplaceValue(1)
        ITK_Vol = segmentationFilter.Execute(ITK_Vol)
        image = self.imageFromITKToNumpy(ITK_Vol)

        return image.astype(np.uint8)

    def imageFromNumpyToITK(self,vol):
        return sitk.GetImageFromArray(vol)

    def imageFromITKToNumpy(self,vol):
        return sitk.GetArrayFromImage(vol)

    def find_center_of_mass(self, array):
        '''
        Args:
            array: array in which to find the center of mass

        Returns:
            center_x = center of mass x-coordinate
            center_y = center of mass y-coordinate
        '''
        # Calculate the mass distribution along the x and y axes
        mass_x = np.sum(array ** 2, axis=0)
        mass_y = np.sum(array ** 2, axis=1)

        # Calculate the center of mass along the x and y axes
        center_x = np.sum(np.arange(array.shape[1]) * mass_x) / np.sum(mass_x)
        center_y = np.sum(np.arange(array.shape[0]) * mass_y) / np.sum(mass_y)
        print(center_x, center_y)
        return center_x, center_y

    def find_angle(self, p1, p2):
        '''
        Args:
            p1: original position
            p2: new position

        Returns: Rotation angle
        '''
        y1 = p1[1]
        y2 = p2[1]

        x1 = p1[0]
        x2 = p2[0]

        alpha = np.arctan((y2 - y1) / (x2 - x1))
        alpha = 180 * alpha / np.pi

        if (x2 - x1) < 0:
            alpha = alpha + 180

        return alpha

    def saveTiff16bit(self, data, filename, minIm=0, maxIm=0, header=None):
        if (minIm == maxIm):
            minIm = np.amin(data)
            maxIm = np.amax(data)
        datatoStore = 65535.0 * (data - minIm) / (maxIm - minIm)
        datatoStore[datatoStore > 65535.0] = 65535.0
        datatoStore[datatoStore < 0] = 0

        datatoStore = np.asarray(datatoStore, np.uint16)


        if (header != None):
            tif.TifImage(data=datatoStore, header=header).write(filename)
        else:
            tif.TifImage(data=datatoStore).write(filename)

    def rotate_volume(self, im_list, alpha, center, output_path):
        '''
        Args:
            im_list: list of images to rotate
            alpha: angle in degrees to be rotated
            center: center of volume
            output_path: path to save the output .npy volume
        '''
        print(output_path)
        vol = np.zeros((len(im_list), 640, 640))
        # Rotate all and create a new dataset
        for i, im in enumerate(im_list):
            print('Image ' + str(i))
            im = Image.open(im)
            f = np.array(im)
            sa = self.shift_array_to_center(f, center)
            vol[i, :, :] = sa
        rot_vol = scipy.ndimage.rotate(vol, alpha, axes =(1,2))
        print('Volume')
        file_name = output_path + 'volume.npy'
        print(file_name)
        np.save(file_name, rot_vol)


if __name__ == '__main__':

    path = 'W:\\Data\\Data_Processing_July2022\\Reconstructions\\'
    out_path = 'W:\\Data\\Data_Processing_July2022\\rot_datasets\\'
    center = (353, 348) # center of the frame
    seg_min_value = 0.3 #minimum threshold value for segmentation
    seg_max_value = 20 #maximum threshold value for segmentation
    threshold = 0.32 #threshold for feature segmentation
    seedListToSegment = [[252, 428, 0], [428, 252, 0]] #seed list give a point to start the region growing segmentation of the battery
    size_median_filter = 5 #size of median filter

    for dataset in os.listdir(path):
        if ('P28A_ISC_FT_H_Exp5_3' in dataset): #selection of datasets
            data_path = path + dataset + '\\'
            path_out = out_path + dataset +'\\'
            if not os.path.isdir(path_out):
                os.mkdir(path_out)
            for j, timestamp in enumerate(os.listdir(data_path)):
                time_path = data_path + timestamp + '\\'
                print(time_path)
                p_out = path_out + timestamp + '\\'
                print(p_out)
                if not os.path.isdir(p_out):
                    os.mkdir(p_out)
                list_images = []
                for i, image in enumerate(os.listdir(time_path)):
                    if ('.tiff' in image):
                        im_path = time_path + image
                        print(im_path)
                        list_images.append(im_path)
                rotation_angle, center = Volume_Rotator.find_rotation(list_images[50], center, max_r, threshold, seedListToSegment, size_median_filter)
                print('Rotating all images')
                Volume_Rotator.rotate_volume(list_images, rotation_angle, center, p_out)
