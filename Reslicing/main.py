import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import fabio.tifimage as tif

class VolumeReslicer:
    '''
    To achieve faster processing of large 3D datasets, the CT volume is horizontally resliced for every degree from 0-180 degrees (see figure).
    '''
    def __init__(self):
        pass

    def move_volume_to_center(self, volume):
        '''
        Args:
            volume: .npy tomographic volume
        Returns: Volume shifted to the center of the frame

        '''
        # Create coordinate grids for each dimension
        x, y, z = np.meshgrid(np.arange(volume.shape[0]),
                              np.arange(volume.shape[1]),
                              np.arange(volume.shape[2]), indexing='ij')

        # Calculate the center of mass using np.sum and normalized by the volume sum
        center_of_mass = np.array([np.sum(x * volume), np.sum(y * volume), np.sum(z * volume)]) / np.sum(volume)

        # Calculate the offset to move the center of mass to the center of the volume
        desired_center = np.array(volume.shape) / 2
        offset = desired_center - center_of_mass

        # Use the offset to create new indices for the shifted volume
        indices = [np.arange(dim_size) - offset[i] for i, dim_size in enumerate(volume.shape)]
        indices = [np.clip(index, 0, dim_size - 1).astype(int) for index, dim_size in zip(indices, volume.shape)]

        # Use integer indices to copy values from the original volume to the new one
        moved_volume = volume[indices[0], :, :]
        moved_volume = moved_volume[:, indices[1], :]
        moved_volume = moved_volume[:, :, indices[2]]

        return moved_volume

    def saveTiff16bit(self,data, filename, minIm=0, maxIm=0, header=None):
        '''
        Args:
            data: np.array data to be saved
            filename: Path for saving
            minIm: minimum pixel value
            maxIm: maximum pixel value
            header:
        '''
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

    def run(self):
        '''
        Function to reslice the tomographic volumes into 2D projections and save each projection as a tiff image.
        '''
        # Define the output path for saving processed images
        output_path = 'INSERT_OUTPUT_PATH_HERE'  # Change this to the desired output path

        # List of paths to input volumes and corresponding starting time numbers
        path_volume_list = ['INSERT_VOLUME_PATHS_HERE']  # Change this to the desired input volume paths
        list_start_t_number = ['START_NUMBERS']  # Change this to the desired starting time numbers related to each path above

        # Iterate over each input volume path and its starting time number
        for path_volume, start_t_number in zip(path_volume_list, list_start_t_number):
            # Extract the name of the volume from the path
            path_name = os.path.basename(os.path.normpath(path_volume))

            # Print the name of the current volume
            print("Processing volume:", path_name)

            # Define the output directory based on the volume name
            output = os.path.join(output_path, path_name)

            # Create the output directory if it does not exist
            if not os.path.isdir(output):
                os.makedirs(output)

            # Iterate over each folder in the input volume directory
            for i, init_folder in enumerate(os.listdir(path_volume)):
                folder = os.path.join(path_volume, init_folder)

                # Iterate over files in the current folder
                for data in os.listdir(folder):
                    print("Processing file:", data)

                    # Load the numpy data as a volume
                    volume_t1 = np.load(data)

                    # Move the volume to center
                    shifted_volume = self.move_volume_to_center(volume_t1)

                    # Extract a region of interest from the shifted volume
                    nvs = shifted_volume[:,152:740, 152:740] #define this region of interest based on the size of your volume

                    # Rotate the region of interest and save resliced images
                    for angle in range(0, 180, 2):
                        rot_vol = scipy.ndimage.rotate(nvs, angle, axes=(1, 2), reshape=False)
                        crop_rect = rot_vol[:, :, int(rot_vol.shape[2] / 2) - 1:int(rot_vol.shape[2] / 2) + 1]
                        comp_rect = np.sum(rot_vol, axis=2) / 3.0
                        out_path = os.path.join(output, init_folder)

                        # Create the output directory if it does not exist
                        if not os.path.isdir(out_path):
                            os.makedirs(out_path)

                        # Save processed image as TIFF
                        self.saveTiff16bit(comp_rect, os.path.join(out_path, str(angle) + '.tiff'))
                        print('Saved tiff')

if __name__ == "__main__":
    image_processor = VolumeReslicer()
    image_processor.run()