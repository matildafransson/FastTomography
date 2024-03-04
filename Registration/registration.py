import SimpleITK as sitk
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import fabio.tifimage as tif


def giveChessImage(Im1, Im2, BlocSpeed):
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


def resample(imageM, imageF, transform):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    interpolator = sitk.sitkCosineWindowedSinc


    return  sitk.Resample(imageM, imageF, transform, interpolator, 0.0,imageM.GetPixelID())


def myshownp(img, title=None, margin=0.05, dpi=80):
    nda = img
    spacing = [1,1]

    ysize = nda.shape[0]
    xsize = nda.shape[1]

    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    fig = plt.figure(title, figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    extent = (0, xsize * spacing[1], 0, ysize * spacing[0])

    t = ax.imshow(nda,
                  extent=extent,
                  interpolation='hamming',
                  cmap='gray',
                  origin='lower')

    if (title):
        plt.title(title)

    plt.show()

def myshowitk(img, title=None, margin=0.05, dpi=80):
    nda = sitk.GetArrayViewFromImage(img)
    spacing = img.GetSpacing()

    ysize = nda.shape[0]
    xsize = nda.shape[1]

    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    fig = plt.figure(title, figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    extent = (0, xsize * spacing[1], 0, ysize * spacing[0])

    t = ax.imshow(nda,
                  extent=extent,
                  interpolation='hamming',
                  cmap='gray',
                  origin='lower')

    if (title):
        plt.title(title)

    plt.show()


def saveTiff16bit(data, filename, minIm=0, maxIm=0, header=None):
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


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


metric_value = []


def command_iteration(method, bspline_transform):
    if method.GetOptimizerIteration() == 0:
        # The BSpline is resized before the first optimizer
        # iteration is completed per level. Print the transform object
        # to show the adapted BSpline transform.
        print(bspline_transform)

    print("{0:3} = {1:10.5f}".format(method.GetOptimizerIteration(),
                                     method.GetMetricValue()))
    metric_value.append(np.log(method.GetMetricValue()))


def command_multi_iteration(method):
    # The sitkMultiResolutionIterationEvent occurs before the
    # resolution of the transform. This event is used here to print
    # the status of the optimizer from the previous registration level.
    if method.GetCurrentLevel() > 0:
        print("Optimizer stop condition: {0}".format(method.GetOptimizerStopConditionDescription()))
        print(" Iteration: {0}".format(method.GetOptimizerIteration()))
        print(" Metric value: {0}".format(method.GetMetricValue()))
        metric_value.append(np.log(method.GetMetricValue()))

    print("--------- Resolution Changing ---------")


def registration_test(itkImageF, itkImageM, output_path):
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

    tx = sitk.BSplineTransformInitializer(image1=itkImageF, transformDomainMeshSize=mesh_size,order=3)

    registration_method.SetInitialTransformAsBSpline(tx)

    registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel([4, 2, 1])

    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method, tx))
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                                   lambda: command_multi_iteration(registration_method))

    outTx = registration_method.Execute(itkImageF, itkImageM)

    print("-------")
    print(tx)
    print(outTx)
    print("Optimizer stop condition: {0}".format(registration_method.GetOptimizerStopConditionDescription()))
    print(" Iteration: {0}".format(registration_method.GetOptimizerIteration()))
    print(" Metric value: {0}".format(registration_method.GetMetricValue()))

    plt.plot(metric_value)
    plt.show()

    FilterTransform = sitk.TransformToDisplacementFieldFilter()
    FilterTransform.SetReferenceImage(itkImageF)
    vectorField = FilterTransform.Execute(outTx)

    filterJacob = sitk.DisplacementFieldJacobianDeterminantFilter()
    deformation_map = filterJacob.Execute(vectorField)
    # deformation_map = sitk.Cast(deformation_map, sitk.sitkFloat32)
    print(deformation_map)
    array = sitk.GetArrayFromImage(deformation_map)
    saveTiff16bit(array, output_path + 'deformation.tiff')
    return outTx

    '''
    #sitk.Show(deformation_map, title="deformation",debugOn=True)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_path+'deformation.tiff')
    writer.Execute(deformation_map)
    '''


if __name__ == '__main__':
    # input_path = '/data/id19/gpfstest/ihma187/id19/c1_100/c1_100_radio'
    # output_path = '/data/projects/whaitiri/Data/c1_100_registration/'
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
    print(mmFilter.GetMinimum(), mmFilter.GetMaximum())
    # sitk.Show(itkImageM, title="cthead1",debugOn=True)


    outTx = registration_test(itkImageM, itkImageF, output_path)

    resampledM = resample(itkImageM, itkImageF, outTx)
    myshowitk(resampledM, 'Resampled Translation')
    chess1 = giveChessImage(sitk.GetArrayFromImage(itkImageF), sitk.GetArrayFromImage(resampledM), 50)
    myshownp(chess1)
