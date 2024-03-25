import SimpleITK as sitk
import os
import numpy as np
import re

class ImageRegistration:
    def __init__(self):
        self.metric_value = []

    def natural_sort(self, l):
        """
        Perform natural sorting on a list.

        Args:
        l (list): List to be sorted.

        Returns:
        list: Sorted list.
        """
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def give_chess_image(self, Im1, Im2, BlocSpeed):
        """
        Generate a chessboard pattern from two images.

        Args:
        Im1 (numpy.ndarray): First image.
        Im2 (numpy.ndarray): Second image.
        BlocSpeed (int): Block speed for chessboard pattern.

        Returns:
        numpy.ndarray: Chessboard pattern.
        """
        SizeXVol = Im1.shape[0]
        SizeYVol = Im1.shape[1]

        iterx = 0

        chessMap = np.zeros((SizeXVol, SizeYVol))

        for xRef in np.arange(0, SizeXVol - 1, BlocSpeed):
            itery = 0
            iterx += 1
            for yRef in np.arange(0, SizeYVol - 1, BlocSpeed):
                itery += 1

                xMinRef = xRef - int(BlocSpeed / 2)
                yMinRef = yRef - int(BlocSpeed / 2)

                xMaxRef = xRef + int(BlocSpeed / 2)
                yMaxRef = yRef + int(BlocSpeed / 2)

                if xMinRef < 0:
                    xMinRef = 0
                if yMinRef < 0:
                    yMinRef = 0

                if xMaxRef >= SizeXVol:
                    xMaxRef = SizeXVol - 1
                if yMaxRef >= SizeYVol:
                    yMaxRef = SizeYVol - 1

                if ((itery + iterx) % 2) == 0:
                    chessMap[xMinRef:xMaxRef, yMinRef:yMaxRef] = Im1[xMinRef:xMaxRef,
                                                                 yMinRef:yMaxRef]
                else:
                    chessMap[xMinRef:xMaxRef, yMinRef:yMaxRef] = Im2[xMinRef:xMaxRef,
                                                                 yMinRef:yMaxRef]

        return chessMap

    def resample(self, imageM, imageF, transform):
        """
        Resample the moving image using the given transformation.

        Args:
        imageM (sitk.Image): Moving image.
        imageF (sitk.Image): Fixed image.
        transform (sitk.Transform): Transformation to be applied.

        Returns:
        sitk.Image: Resampled image.
        """
        interpolator = sitk.sitkCosineWindowedSinc
        return sitk.Resample(imageM, imageF, transform, interpolator, 0.0, imageM.GetPixelID())

    def command_iteration(self, method, bspline_transform):
        """
        Callback function executed on each iteration of the optimizer.

        Args:
        method (sitk.ImageRegistrationMethod): Image registration method.
        bspline_transform (sitk.BSplineTransform): BSpline transformation.
        """
        if method.GetOptimizerIteration() == 0:
            print(bspline_transform)

        print("{0:3} = {1:10.5f}".format(method.GetOptimizerIteration(), method.GetMetricValue()))
        self.metric_value.append(np.log(method.GetMetricValue()))

    def registration(self, fixed_image, moving_image):
        """
        Perform image registration.

        Args:
        fixed_image (str): Path to the fixed image.
        moving_image (str): Path to the moving image.

        Returns:
        tuple: Tuple containing fixed, moving images, vector field, final transform, and deformation map.
        """
        fixed = sitk.ReadImage(fixed_image, sitk.sitkFloat32)
        moving = sitk.ReadImage(moving_image, sitk.sitkFloat32)

        # Additional registration code here

        return fixed, moving, vectorField, final_transform, deformation_map

if __name__ == '__main__':
    registration_obj = ImageRegistration()
    directory = 'W:\\Data\\Data_Processing_July2022\\reslicedData\\'
    out_directory = 'W:\\Data\\Data_Processing_July2022\\Displacement_arrays\\'

    for experiment in os.listdir(directory):
        out_exp_path = out_directory + experiment
        if not os.path.isdir(out_exp_path):
            os.mkdir(out_exp_path)

        vector_path = out_exp_path + '\\' + 'VectorField'
        if not os.path.isdir(vector_path):
            os.mkdir(vector_path)

        transform_path = out_exp_path + '\\' + 'Transform'
        if not os.path.isdir(transform_path):
            os.mkdir(transform_path)

        deformation_path = out_exp_path + '\\' + 'Deformation'
        if not os.path.isdir(deformation_path):
            os.mkdir(deformation_path)

        if ('VCT5A_FT_H_Exp5' in experiment):
            exp_path = directory + experiment + '\\'
            list_t_entry = []
            for t_entry in os.listdir(exp_path):
                if not('nabu_cfg_files' in t_entry):
                    t_entry_path = exp_path + t_entry
                    list_t_entry.append(t_entry_path)
            list_t_entry_crop = list_t_entry[::2]
            len_list = len(list_t_entry_crop)

            for i in range(0,len_list,1):
                if i < len_list-1:
                    entry1 = list_t_entry_crop[i]
                    name_1 = entry1.split('\\')[-1]
                    name_1 = name_1.split('_')[0]
                    entry2 = list_t_entry_crop[i+1]
                    name_2 = entry2.split('\\')[-1]
                    name_2 = name_2.split('_')[0]
                    list_t1_angle = []
                    list_t2_angle = []
                    for t1_angle, t2_angle in zip(os.listdir(entry1),os.listdir(entry2)):
                        t1_angle_path = entry1 + '\\' + t1_angle
                        t2_angle_path = entry2 + '\\' + t2_angle
                        list_t1_angle.append(t1_angle_path)
                        list_t2_angle.append(t2_angle_path)
                    sorted_t1 = registration_obj.natural_sort(list_t1_angle)
                    sorted_t2 = registration_obj.natural_sort(list_t2_angle)
                    for tiff1, tiff2 in zip(sorted_t1,sorted_t2):

                        path_im_1 = tiff1
                        tiff_name_1 = path_im_1.split('\\')[-1]
                        tiff_name_1 = tiff_name_1.split('.')[0]
                        path_im_2 = tiff2
                        tiff_name_2 = path_im_2.split('\\')[-1]
                        tiff_name_2 = tiff_name_2.split('.')[0]