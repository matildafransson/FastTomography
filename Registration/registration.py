import SimpleITK as sitk
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import fabio.tifimage as tif

class ImageRegistration:
    def __init__(self):
        self.metric_value = []

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

    def command_multi_iteration(self, method):
        """
        Callback function executed on each resolution level change.

        Args:
        method (sitk.ImageRegistrationMethod): Image registration method.
        """
        if method.GetCurrentLevel() > 0:
            print("Optimizer stop condition: {0}".format(method.GetOptimizerStopConditionDescription()))
            print(" Iteration: {0}".format(method.GetOptimizerIteration()))
            print(" Metric value: {0}".format(method.GetMetricValue()))
            self.metric_value.append(np.log(method.GetMetricValue()))

        print("--------- Resolution Changing ---------")

    def registration_test(self, itkImageF, itkImageM, output_path):
        """
        Perform image registration.

        Args:
        itkImageF (sitk.Image): Fixed image.
        itkImageM (sitk.Image): Moving image.
        output_path (str): Path to save the results.

        Returns:
        sitk.Transform: Output transformation.
        """
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMeanSquares()
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.02)
        registration_method.SetInterpolator(sitk.sitkBSpline)
        registration_method.SetOptimizerAsGradientDescent(learningRate=0.001, numberOfIterations=1000,
                                                          convergenceMinimumValue=1e-6, convergenceWindowSize=10)

        x_grid_size = 500
        y_grid_size = 500
        grid_physical_spacing = [x_grid_size, y_grid_size]
        image_physical_size = [size * spacing for size, spacing in zip(itkImageF.GetSize(), itkImageF.GetSpacing())]
        mesh_size = [int(image_size / grid_spacing + 0.5) for image_size, grid_spacing in
                     zip(image_physical_size, grid_physical_spacing)]

        tx = sitk.BSplineTransformInitializer(image1=itkImageF, transformDomainMeshSize=mesh_size, order=3)
        registration_method.SetInitialTransformAsBSpline(tx)
        registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel([4, 2, 1])

        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: self.command_iteration(registration_method, tx))
        registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                                       lambda: self.command_multi_iteration(registration_method))

        outTx = registration_method.Execute(itkImageF, itkImageM)

        print("-------")
        print(tx)
        print(outTx)
        print("Optimizer stop condition: {0}".format(registration_method.GetOptimizerStopConditionDescription()))
        print(" Iteration: {0}".format(registration_method.GetOptimizerIteration()))
        print(" Metric value: {0}".format(registration_method.GetMetricValue()))

        plt.plot(self.metric_value)
        plt.show()

        filter_transform = sitk.TransformToDisplacementFieldFilter()
        filter_transform.SetReferenceImage(itkImageF)
        vector_field = filter_transform.Execute(outTx)

        filter_jacobian = sitk.DisplacementFieldJacobianDeterminantFilter()
        deformation_map = filter_jacobian.Execute(vector_field)

        array = sitk.GetArrayFromImage(deformation_map)
        self.save_tiff_16bit(array, output_path + 'deformation.tiff')

        return outTx

    def save_tiff_16bit(self, data, filename, minIm=0, maxIm=0, header=None):
        """
        Save data as 16-bit TIFF.

        Args:
        data (np.ndarray): Image data.
        filename (str): Output filename.
        minIm (float): Minimum intensity value.
        maxIm (float): Maximum intensity value.
        header (str): Header information.
        """
        if minIm == maxIm:
            minIm = np.amin(data)
            maxIm = np.amax(data)
        datatoStore = 65535.0 * (data - minIm) / (maxIm - minIm)
        datatoStore[datatoStore > 65535.0] = 65535.0
        datatoStore[datatoStore < 0] = 0
        datatoStore = np.asarray(datatoStore, np.uint16)

        if header is not None:
            tif.TifImage(data=datatoStore, header=header).write(filename)
        else:
            tif.TifImage(data=datatoStore).write(filename)

def natural_sort(l):
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

if __name__ == '__main__':
    input_path = 'Z:\\gpfstest\\ihma187\\id19\\c1_100\\c1_100_radio'
    output_path = 'W:\\Data\\c1_100_registration\\'

    list_prj_images = natural_sort(os.listdir(input_path))

    path_first_image = input_path + '/' + list_prj_images[50000]
    path_second_image = input_path + '/' + list_prj_images[75000]
    itkImageF = sitk.ReadImage(path_first_image)
    itkImageF = sitk.Cast(itkImageF, sitk.sitkFloat32)
    itkImageM = sitk.ReadImage(path_second_image)
    itkImageM = sitk.Cast(itkImageM, sitk.sitkFloat32)

    mmFilter = sitk.MinimumMaximumImageFilter()
    mmFilter.Execute(itkImageM)
