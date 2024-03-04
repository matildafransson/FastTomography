<u>**CT Reconstruction:**</u>

Fast tomography of battery failure is performed using a specialized synchrotron-adapted setup. 
The incorporated rotation stage and the high-speed detector that enables fast CT, are not directly linked. 
To overcome this issue and enable CT reconstruction, the rotation angles are accessed using a fast 
Transcom recording of the motor positions. The angular positions are then synced with the projections 
and saved in a NeXus (.nx) file format, such that each projection is attributed with the correct rotation angle. 
The angles are recorded and saved in a .txt file, that in the first part of the code is read and cut to match 
the projections (.tiff files). Codes for this part are found in the folder Reconstruction. 

In the Reconstruction main.py, please provide:

main_folder = the main folder to data. Later on, it is possible to loop through several experiments or select specific experiments to be processed, therefore provide only the main path here.

path_dark = path to where the dark images are located 

path_ref = path to where the reference images are located 

flag360 = True for reconstruction every 360 degrees, and False for reconstruction every 180 degrees. 

pixel_size = Give the pixel size in [m]

energy = Give the X-ray energy in [keV]

distance = Give the sample/detector distance in [m]

In the following step, CT reconstruction is done from the .nx file using the ESRF-developed software 
(find documentation here: (https://gitlab.esrf.fr/tomotools/nabu) to obtain the 3D volumes. 
This algorithm pre-processes the projections through normalization using flat and dark-field images,
then reconstructs using the FBP-reconstruction algorithm incorporated in Nabu. Paganing filtering 
is applied in the reconstruction step to increase the image quality.


<u>**Data Rotation:**</u>

To ensure that all batteries have the same orientation, the volumes have been automatically rotated 
based on a feature location, and afterwards saved as .npy files for easier post-processing. 

Processing of reconstructed and rotated data is thereafter followed by 'Speed Retrieval'
or 'Tracking of Metal Agglomerate Segmentation'. 

<u>**Metal Agglomerate Segmentation:**</u>

Code for this has been developed by Matteo Venturelli and can be found in this Git repository: 

<u>**Speed-Retieval:**</u> 

'![Figure 1: Process of speed retrieval from fast tomography data](https://github.com/matildafransson/FastTomography/blob/master/FINAL_SPEED_FIG.png?raw=true)'

**A. Reslicing**

In a primary step to achieve faster processing of large 3D datasets, the CT volume is horizontally resliced for every degree from 0-180. See step A in the figure. 

**B. Registration**

**C+D. Speed Mapping and Plotting**


 

Resliced datasets from two subsequent time steps at the same angular positions are thereafter considered in pairs for the registration process, see step B in the figure.
