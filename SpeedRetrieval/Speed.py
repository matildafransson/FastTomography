import os
import re
import numpy as np
from matplotlib import pyplot as plt
import skimage
import math
from scipy.stats import ranksums
from matplotlib import cm

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

class VectorAnalysis():

    def __init__(self, path_vector, time_step, angle_step, path_save_exp):
        self.time_step = time_step
        self.path_vector = path_vector
        self.angle_step = angle_step
        self.path_save_exp = path_save_exp
        self.path_save = '\\'.join(self.path_vector.split('\\')[:-1]) + '\\'
        self.npy_file_path = 'Plot_Array.npy'
        self.npy_file_path_v = 'Plot_Array_v.npy'
        self.npy_file_path_plot = 'Plot_Array_Plot.npy'
    def save_data(self):
        if not self.npy_file_path in os.listdir(self.path_save):
            np.save(self.path_save + self.npy_file_path, self.data)

    def save_data_plot(self):
        if not self.npy_file_path_plot in os.listdir(self.path_save):
            np.save(self.path_save + self.npy_file_path_plot, self.data_plot)

    def save_data_v(self):
        if not self.npy_file_path_v in os.listdir(self.path_save):
            np.save(self.path_save + self.npy_file_path_v, self.data_v)

    def load_data(self):
        if not self.npy_file_path in os.listdir(self.path_save):
            list_files = []
            for t_angle_file in os.listdir(self.path_vector):
                file_path = self.path_vector + '\\' + t_angle_file
                list_files.append(file_path)
            list_files = natural_sort(list_files)
            nr_time = int(len(list_files) / 90)
            nr_angles = 180

            self.data = np.zeros((nr_time,nr_angles,2))

            for i, file in enumerate(list_files):
                angle1 = int(((i * 2) % 180)/2)
                angle2 = angle1 + 90
                time = i//90
                array = np.load(file)
                middle_row_index = array.shape[1] // 2
                #splitting the data in 2
                angle1_array = array[:,:middle_row_index, :]
                angle2_array = array[:, middle_row_index:,:]
                angle1_array_u = np.average(angle1_array[:,:,0])
                angle1_array_v = np.average(angle1_array[:,:,1])
                angle2_array_u = np.average(angle2_array[:, :, 0])
                angle2_array_v = np.average(angle2_array[:, :, 1])

                self.data[time, angle1, 0] = angle1_array_u
                self.data[time, angle1, 1] = angle1_array_v
                self.data[time, angle2, 0] = angle2_array_u
                self.data[time, angle2, 1] = angle2_array_v
                print(time, angle1, angle2)

            self.save_data()
        else:
            self.data = np.load(self.path_save + self.npy_file_path)

    def load_data_v(self):
        if not self.npy_file_path_v in os.listdir(self.path_save):
            list_files = []
            for t_angle_file in os.listdir(self.path_vector):
                file_path = self.path_vector + '\\' + t_angle_file
                list_files.append(file_path)
            list_files = natural_sort(list_files)

            nr_x = np.load(list_files[0]).shape[0]
            nr_time = int(len(list_files) / 90)

            self.data_v = np.zeros((nr_time, nr_x))

            for i, file in enumerate(list_files):
                time = i // 90
                angle1 = int(((i * 2) % 180) / 2)
                array = np.load(file)[:,:,1]
                array = np.mean(array, axis = 1)
                self.data_v[time,:] += (array/90.0)
            self.save_data_v()
        else:
            self.data_v = np.load(self.path_save + self.npy_file_path_v)

    def load_plot_data(self):
        if not self.npy_file_path_plot in os.listdir(self.path_save):
            list_files = []
            for t_angle_file in os.listdir(self.path_vector):
                file_path = self.path_vector + '\\' + t_angle_file
                list_files.append(file_path)
            list_files = natural_sort(list_files)

            nr_time = int(len(list_files) / 90)
            nr_angles = 180

            self.data_plot = np.zeros((nr_time, nr_angles, 3))

            for i, file in enumerate(list_files):
                angle1 = int(((i * 2) % 180)/2)
                time = i//90
                array = (np.abs(np.load(file)))
#--------------------------------------------------------------------------
                sec_o_1_u = np.average(array[89:166, :, 0])
                if math.isnan(sec_o_1_u):
                    sec_o_1_u = 0.0

                sec_o_2_u = np.average(array[474:551, :, 0])
                if math.isnan(sec_o_2_u):
                    sec_o_2_u = 0.0

                sec_o_1_v = np.average(array[89:166, :, 1])
                if math.isnan(sec_o_1_v):
                    sec_o_1_v = 0.0

                sec_o_2_v = np.average(array[474:551, :, 1])
                if math.isnan(sec_o_2_v):
                    sec_o_2_v = 0.0

                speed_sec_o_1 = np.sqrt(sec_o_1_u ** 2 + sec_o_1_v ** 2)
                speed_sec_o_2 = np.sqrt(sec_o_2_u ** 2 + sec_o_2_v ** 2)

                av_speed_sec_o = (speed_sec_o_1 + speed_sec_o_2)/2
                self.data_plot[time, angle1, 0] = av_speed_sec_o
#------------------------------------------------------------------------------------

                sec_m_1_u = np.average(array[166:243, :, 0])
                if math.isnan(sec_m_1_u):
                    sec_m_1_u = 0.0

                sec_m_2_u = np.average(array[397:474, :, 0])
                if math.isnan(sec_m_2_u):
                    sec_m_2_u = 0.0

                sec_m_1_v = np.average(array[166:243, :, 1])
                if math.isnan(sec_m_1_v):
                    sec_m_1_v = 0.0

                sec_m_2_v = np.average(array[397:474, :, 1])
                if math.isnan(sec_m_2_v):
                    sec_m_2_v = 0.0

                speed_sec_m_1 = np.sqrt(sec_m_1_u ** 2 + sec_m_1_v ** 2)
                speed_sec_m_2 = np.sqrt(sec_m_2_u ** 2 + sec_m_2_v ** 2)

                av_speed_sec_m = (speed_sec_m_1 + speed_sec_m_2)/2
                self.data_plot[time, angle1, 1] = av_speed_sec_m

#-------------------------------------------------------------------------------

                sec_c_u = np.average(array[243:397, :, 0])
                if math.isnan(sec_c_u):
                    sec_c_u = 0.0

                sec_c_v = np.average(array[243:397, :, 1])
                if math.isnan(sec_m_2_v):
                    sec_c_v = 0.0

                speed_sec_c = np.sqrt(sec_c_u ** 2 + sec_c_v ** 2)
                self.data_plot[time, angle1, 2] = speed_sec_c

            self.save_data_plot()
        else:
            self.data_plot = np.load(self.path_save + self.npy_file_path_plot)


    def speed_conversion(self):
        self.data = np.abs(self.data * 0.8)
        self.data_v = np.abs(self.data_v * 0.8)
        self.data_plot = self.data_plot * 0.8

    def reduce_time_res(self):
        new_time_shape = int(np.floor(self.data.shape[0] /self.time_step))
        self.data = self.data[:new_time_shape*self.time_step,:, :]
        self.reduced_t_data = skimage.transform.downscale_local_mean(self.data,(self.time_step,1,1))

    def reduce_angle_res(self):
        self.reduced_angle_t_data = skimage.transform.downscale_local_mean(self.reduced_t_data, (1, self.angle_step, 1))

    def std_calculation(self):
        self.std_data = np.std(self.data, axis=0)
        self.std_data_angle = skimage.transform.downscale_local_mean(self.std_data,(self.angle_step,1))

    def mean_calculation(self):
        self.mean_data_angle = np.mean(self.reduced_angle_t_data,axis =0)
        self.mean_data = np.mean(self.reduced_t_data, axis=0)

    def v_mean_calculation(self):
        self.mean_data_v = np.mean(self.data_v,axis = 0 )

    def v_std_calculation(self):
        self.std_data_v = np.std(self.data_v, axis=0)

    def plot_mean_calculation(self):
        self.mean_data_plot = np.mean(self.data_plot, axis = 1)

    def ten_percent_calculation(self):
        self.angles = np.arange(0,362,2)
        self.angles_calc = np.radians(self.angles)
        self.mean_data = np.append(self.mean_data, [[self.mean_data[0, 0], self.mean_data[0, 1]]], axis=0)
        self.speeds = self.mean_data[:, 0]

        # Calculate the 10% threshold for highest speeds
        self.threshold_speed = np.percentile(self.speeds, 90)

        # Select data points with speeds above the threshold
        self.high_speed_indices = np.where(self.speeds >= self.threshold_speed)[0]
        self.high_speed_angles = self.angles_calc[self.high_speed_indices]

        # Calculate the circular mean of high-speed angles
        mean_high_speed_angle = np.angle(np.mean(np.exp(1j * self.high_speed_angles)))

        # Determine the angle range for analysis
        self.angle_range_start = mean_high_speed_angle - np.pi / 4  # Adjust the range as needed
        self.angle_range_end = mean_high_speed_angle + np.pi / 4

        # Ensure the angle range is within [0, 2*pi)
        self.angle_range_start = (self.angle_range_start + 2 * np.pi) % (2 * np.pi)
        self.angle_range_end = (self.angle_range_end + 2 * np.pi) % (2 * np.pi)

        # Select speeds within the specified angle range
        self.speeds_in_range = self.speeds[(self.angles_calc >= self.angle_range_start) & (self.angles_calc <= self.angle_range_end)]

        # Perform the Wilcoxon rank-sum test
        statistic, p_value = ranksums(self.speeds_in_range, self.speeds)

        # Print the result
        if p_value < 0.05:
            self.result_text = "Significant"
            self.significant = True
        else:
            self.result_text = "Homogeneous"
            self.significant = False

        # Print the calculated angle range and p-value
        self.angle_range_text = f"Angle Range: {int(np.degrees(self.angle_range_start)):.2f}° - {int(np.degrees(self.angle_range_end)):.2f}°"
        self.p_value_text = f"P-value: {p_value:.4f}"
        print('Radial Average :' , np.mean(self.speeds))
        print('Vertical Average :' ,np.mean(self.mean_data_v))


    def display(self):
        plt.rcParams.update({'font.size': 12 })
        viridis_colormap = plt.get_cmap('viridis')

        self.angles_reduced = np.arange(0,390,30)
        self.std_data = np.append(self.std_data,[[self.std_data[0,0],self.std_data[0,1]]],axis=0)

        plt.figure()
        c1 = plt.polar(np.radians(self.angles), self.mean_data[:,0]+self.std_data[:,0], color= viridis_colormap(0.6), alpha = 0.7, linestyle='dashed', label = 'Mean speed + STD [mm/s]')[0] #'#3f8502'
        c1.set_theta_zero_location("W")
        x1 = c1.get_xdata()
        y1 = c1.get_ydata()
        c2 = plt.polar(np.radians(self.angles), self.mean_data[:,0] - self.std_data[:,0], color= viridis_colormap(0.9),alpha = 0.7, linestyle='dashed', label = 'Mean speed - STD [mm/s]')[0] #'#00735d'
        x2 = c2.get_xdata()
        y2 = c2.get_ydata()
        plt.fill_between(x1, y2, y1, color=viridis_colormap(0.3), alpha = 0.4)
        plt.polar(np.radians(self.angles), self.mean_data[:, 0], color=viridis_colormap(0.3), alpha = 1, linestyle='-', label = 'Mean speed [mm/s]') #'#f7712d'
        color1 = cm.viridis(0.3)
        color2 = cm.viridis(0.9)
        ax = plt.gca()
        ax.scatter(self.angles, self.speeds, c=color1, s=50, alpha=0.5, label='All Speeds')
        ax.scatter(self.high_speed_angles, self.speeds[self.high_speed_indices], c=color2, s=80, alpha=0.7, label='Max 10% Speeds')

        if self.significant:
            angles = np.radians(np.linspace(0, 360, 360))
            radii = np.linspace(-1, 1, 360)

            start_angle = np.degrees(self.angle_range_start)
            end_angle = np.degrees(self.angle_range_end)
            outer_ring_radii = radii[-1] * np.ones_like(angles)
            min_radial_value = np.min(self.mean_data[:,0] - self.std_data[:,0])

            # Choose the color for the cone
            cone_color = 'green'

            # Plot the colored cone using fill_between
            highlight_label = self.angle_range_text
            plt.fill_between(angles,min_radial_value,outer_ring_radii,
                            where=(angles >= np.radians(start_angle)) & (angles <= np.radians(end_angle)),
                            color=cone_color, alpha=0.08, label = highlight_label)

        ax.yaxis.grid(linestyle='dashed')
        ax.xaxis.grid(linestyle = 'dashed')
        plt.title(self.path_vector.split('\\')[-2])
        width_cm = 30  # Specify the width in centimeters
        height_cm = 30  # Specify the height in centimeters

        # Convert centimeters to inches
        width_inches = width_cm * 0.393701
        height_inches = height_cm * 0.393701
        plt.gcf().set_size_inches(width_inches, height_inches)
        existing_legend = plt.legend()
        text1 = self.p_value_text
        text2 = self.result_text
        print(self.result_text)
        print(self.p_value_text)

        # Add the existing legend
        legend_text = ax.text(1.25, 1, f'{text1}\n{text2}', transform=ax.transAxes,
                              verticalalignment='center', bbox=dict(facecolor='white', alpha=0.7))


        plt.savefig(path_save_exp + '_output_plot_rad.pdf', format='pdf')
        plt.savefig(path_save_exp + '_output_plot_rad.png', format='png')
        plt.show()

        x = np.arange(-self.data_v.shape[1]/2,self.data_v.shape[1]/2, 1) * 40/10**3
        plt.figure()
        plt.plot(x, self.mean_data_v + self.std_data_v,  color =  viridis_colormap(0.1) , alpha = 0.7,linestyle='dashed', label = 'Mean speed + STD') #'#0088ff'
        plt.plot(x, self.mean_data_v - self.std_data_v,  color = viridis_colormap(0.4), alpha = 0.7,linestyle='dashed', label = 'Mean speed - STD') #'#00c7fa'
        plt.fill_between(x, self.mean_data_v + self.std_data_v, self.mean_data_v - self.std_data_v, color = viridis_colormap(0.1), alpha = 0.2) #'#0088ff'
        plt.plot(x, self.mean_data_v, color = viridis_colormap(1), alpha = 1, linestyle='-',label = 'Mean speed')#'#0922DB'
        plt.grid(linestyle='dashed')
        # Hide the right and top spines
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        arrow_properties = dict(arrowstyle='->', lw=2, color='gray')
        plt.title(self.path_vector.split('\\')[-2])
        plt.legend()
        plt.xlabel('[mm]')
        plt.ylabel('[mm/s]')
        width_cm = 30  # Specify the width in centimeters
        height_cm = 20  # Specify the height in centimeters

        # Convert centimeters to inches
        width_inches = width_cm * 0.393701
        height_inches = height_cm * 0.393701
        plt.gcf().set_size_inches(width_inches, height_inches)
        plt.savefig(path_save_exp + '_output_plot_z.pdf', format='pdf')
        plt.savefig(path_save_exp + '_output_plot_z.png', format='png')
        plt.show()

        t_array = np.arange(0,self.mean_data_plot.shape[0], 1)/20
        print(t_array)
        print(len(t_array))
        print(self.mean_data_plot[:,0].shape)
        plt.plot(t_array, self.mean_data_plot[:,0], label = 'External', color = viridis_colormap(0.6) )
        plt.plot(t_array,self.mean_data_plot[:,1], label = 'Intermediate', color = viridis_colormap(0.9))
        plt.plot(t_array, self.mean_data_plot[:,2], label = 'Core', color = viridis_colormap(0.3))
        plt.title(self.path_vector.split('\\')[-2])
        plt.legend()
        plt.grid(True, linestyle='--')
        plt.ylabel('Delamination Speed [mm/s]')
        plt.xlabel('Time [s]')
        width_cm = 30  # Specify the width in centimeters
        height_cm = 20  # Specify the height in centimeters

        # Convert centimeters to inches
        width_inches = width_cm * 0.393701
        height_inches = height_cm * 0.393701
        plt.gcf().set_size_inches(width_inches, height_inches)
        plt.savefig(path_save_exp + '_output_plot_sections_speed.png', format='png')
        plt.show()

    def printing(self):
        max_speed_index = np.argmax(self.mean_data[:, 0])
        min_speed_index = np.argmin(self.mean_data[:, 0])

        # Retrieve the corresponding angles for max and min speeds
        angle_for_max_speed = self.angles[max_speed_index]
        angle_for_min_speed = self.angles[min_speed_index]

        # Get the actual values for max and min speeds
        max_speed = self.mean_data[:, 0][max_speed_index]
        min_speed = self.mean_data[:, 0][min_speed_index]

        print("Maximum Speed:", max_speed, "at Angle:", angle_for_max_speed)
        print("Minimum Speed:", min_speed, "at Angle:", angle_for_min_speed)

        max_speed_v = np.max(self.mean_data_v)
        min_speed_v = np.min(self.mean_data_v)

        print("Maximum Speed v:", max_speed_v)
        print("Minimum Speed v:", min_speed_v)

# Comment
if __name__ == '__main__':

    path = #Insert data path
    path_save_fig = # Insert path where to save images
    time_steps = #Select the number of time steps you want to consider while computing the averages
    angle_step = #select the angular interval for the computation
    list_experiments = []
    save_path_list = []
    for experiment in os.listdir(path):
        if 'VCT5A_FT_H_Exp4' in experiment:
            exp_path= path + experiment
            save_path = path_save_fig + experiment
            save_path_list.append(save_path)
            for folder in os.listdir(exp_path):
                if 'VectorField' in folder:
                    vf_path = exp_path + '\\' + folder
                    list_experiments.append(vf_path)


    for path_save_exp, exp in zip(save_path_list,list_experiments):
        analysis = VectorAnalysis(exp, time_steps, angle_step, path_save_exp)
        analysis.load_data()
        analysis.load_data_v()
        analysis.load_plot_data()
        analysis.speed_conversion()
        analysis.reduce_time_res()
        analysis.reduce_angle_res()
        analysis.std_calculation()
        analysis.v_std_calculation()
        analysis.mean_calculation()
        analysis.v_mean_calculation()
        analysis.plot_mean_calculation()
        analysis.ten_percent_calculation()
        analysis.display()
        analysis.printing()



