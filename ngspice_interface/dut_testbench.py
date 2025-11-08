import time
import numpy as np
import os
import scipy.interpolate as interp
import scipy.optimize as sciopt
from scipy.optimize import differential_evolution
from .area_estimation import BPTM45nmAreaEstimator
from .ngspice_wrapper import NgspiceWrapper
import matplotlib.pyplot as plt


class DUT(NgspiceWrapper):
    def measure_metrics(self):
        self.parse_outputs()
        spec_dict = {}
        # post process raw data
        area_estimator = BPTM45nmAreaEstimator(self.circuit_params, self.circuit_multipliers)
        spec_dict['area'] = area_estimator.find_area()
        spec_dict['current'] = self.current
        spec_dict['gain'] = self.find_dc_gain(self.vout_complex)
        spec_dict['noise'] = self.noise
        spec_dict['phm'] = self.find_phm(self.freq, self.vout_complex)
        spec_dict['slewRate'] = self.find_slew_rate(self.time, self.vout_tran, threshold_low=0.1, threshold_high=0.9, time_unit='us')
        spec_dict['ugbw'] = self.find_ugbw(self.freq, self.vout_complex)

        return spec_dict
    
    def parse_outputs(self):
        tran_fname = os.path.join(self.output_files_folder, 'tran_'+self.random_name+'.csv')
        ac_fname = os.path.join(self.output_files_folder, 'ac_'+self.random_name+'.csv')
        dc_fname = os.path.join(self.output_files_folder, 'dc_'+self.random_name+'.csv')
        noise_fname = os.path.join(self.output_files_folder, 'noise_'+self.random_name+'.csv')
        # add these file names in a list
        self.output_files = [tran_fname, ac_fname, dc_fname, noise_fname]
        for file in self.output_files:
            if not os.path.isfile(file):
                print(f"{file} doesn't exist")
        tran_raw_outputs = np.genfromtxt(tran_fname, skip_header=1)
        ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=1)
        dc_raw_outputs = np.genfromtxt(dc_fname, skip_header=1)
        noise_raw_outputs = np.genfromtxt(noise_fname, skip_header=1)

        self.time = tran_raw_outputs[:, 0]
        self.vout_tran = tran_raw_outputs[:, 1]
        
        self.freq = ac_raw_outputs[:, 0]
        vout_real = ac_raw_outputs[:, 1]
        vout_imag = ac_raw_outputs[:, 2]
        self.vout_complex = vout_real + 1j*vout_imag
        
        self.current = - dc_raw_outputs[1]

        self.noise = noise_raw_outputs[0]
    
    def find_dc_gain(self, vout):
        """
        TODO: Implement the DC gain calculation
        
        Hint:
        Use numpy's abs() function to calculate the magnitude of the complex number at each point.
        """
        # Vin's amp is 1
        magnitude = np.abs(vout)
    
        dc_gain_linear = magnitude[0]   
        # dc_gain_db = 20 * np.log10(dc_gain_linear)
        
        return dc_gain_linear
        pass
        
    def find_ugbw(self, freq, vout):
        """
        TODO: Implement the unity gain bandwidth (UGBW) calculation
        
        Hints:
        1. Calculate the magnitude of vout
        2. Find where the magnitude crosses 1 (unity gain)
        3. Use _get_best_crossing() to find the crossing point through interpolation
        4. What should you if no crossing is found? What situations can lead to this?
        """
        magnitude  = np.abs(vout)
        ubgw, found = self._get_best_crossing(freq,magnitude,1)
        if found:
            print("ubgw found at: ", ubgw)
            return ubgw
        else:
            if magnitude[0]>1:
                print("Warning: no unity-gain crossing found, measure range is not enough")
            else: 
                print("warning: gain always lower than 1")
            return -1
        pass
    
    def find_phm(self, freq, vout):
        """
        TODO: Implement the phase margin (PHM) calculation
        
        Hints:
        1. Calculate gain array and phase array from vout
        2. Find the unity gain frequency (UGBW)
        3. Interpolate to find the phase at UGBW (you can use interp.interp1d quadratic interpolation)
        4. Calculate phase margin (watch out for radians/degrees units and phase wrap around)
        5. Handle edge cases (e.g., when gain is always < 1) --> hint: you can think in RL terms; worst case reward ...
        """
        gain = np.abs(vout)
        phase = np.angle(vout, deg=True)
        ugbw = self.find_ugbw(freq,vout)
        if ugbw == -1:
            print("warning, no ugbw found, phm not defined")
            return 0
        phase_fun = interp.interp1d(freq, phase, kind='quadratic', fill_value='extrapolate')
        phase_at_ugbw = float(phase_fun(ugbw))
        phm = 180.0 + phase_at_ugbw
        print("phm found:", phm)
        return phm

        pass
    
    def find_slew_rate(self, time, signal, threshold_low=0.1, threshold_high=0.9, time_unit='us'):
        """
        TODO: Implement the slew rate calculation
        
        Hints:
        1. Find large rising edges in the signal
        2. Calculate slope for each rising edge
        3. Take the average of these slopes
        5. Handle edge cases (e.g., no rising edges found)
        6. Final value should be in V/us
        """
        plt.plot(time, signal)
        plt.title("Input Signal for Slew Rate Calculation")
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        plt.grid(True)
        plt.show()
        v_min, v_max = max(np.min(signal),0), min(np.max(signal),1.20)
        v_low = v_min + threshold_low * (v_max - v_min)
        v_high = v_min + threshold_high * (v_max - v_min)
        v_diff = v_high - v_low
        slew_rates = []
        i = 0
        n = len(signal)
        while i < n-1:
            while i < n - 1 and signal[i] > v_low:
                i += 1
            # Now signal[i] <= v10
            # while i < n - 1 and signal[i+1] < v_low:
            #     i += 1
            start_idx = i

            # Find when it next exceeds v90
            while i < n - 1 and signal[i] < v_high:
                i += 1
            stop_idx = i

            if stop_idx-start_idx >=4 and stop_idx < n:
                time_seg = time[start_idx:stop_idx+1]
                sig_seg = signal[start_idx:stop_idx+1]
                t_low, low_found = self._get_best_crossing(time_seg,sig_seg,v_low)
                t_high, high_found = self._get_best_crossing(time_seg,sig_seg,v_high)
                if low_found and high_found:
                    slope = v_diff/(t_high-t_low)
                    slew_rates.append(slope)
                    i+=1   
        if len(slew_rates) == 0:
            print("warning: no big edge is found")
            return 0

        avg_slew = np.mean(slew_rates)

        return avg_slew*1e-6   #unit should be V/us

        pass

    def _get_best_crossing(cls, xvec, yvec, val):
        interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        def fzero(x):
            return interp_fun(x) - val

        xstart, xstop = xvec[0], xvec[-1]
        try:
            return sciopt.brentq(fzero, xstart, xstop), True
        except ValueError:
            return xstop, False
        
if __name__ == "__main__":
    project_path = os.getcwd()
    yaml_path = os.path.join(project_path, 'ngspice_interface', 'files', 'yaml_files', 'TwoStage.yaml')
    parameters_0 = {
        'mp1': 10,
        'wp1': 5.0e-07,
        'lp1': 100.0e-09,
        'mn1': 10,
        'wn1': 5.0e-07,
        'ln1': 100.0e-09,
        'mp3': 10,
        'wp3': 5.0e-07,
        'lp3': 100.0e-09,
        'mn3': 10,
        'wn3': 5.0e-07,
        'ln3': 100.0e-09,
        'mn4': 10,
        'wn4': 5.0e-07,
        'ln4': 100.0e-09,
        'mn5': 10,
        'wn5': 5.0e-07,
        'ln5': 100.0e-09,
        'cap': 5.0e-12,
        'res': 5.0e+3
    }
    parameters_1 = {
    'mp1': 6,
    'wp1': 2.25e-06,
    'lp1': 1.35e-07,
    'mn1': 10,
    'wn1': 5e-07,
    'ln1': 1.35e-07,
    'mp3': 10,
    'wp3': 1.75e-06,
    'lp3': 9e-08,
    'mn3': 1,
    'wn3': 1e-06,
    'ln3': 1.35e-07,
    'mn4': 9,
    'wn4': 1.25e-06,
    'ln4': 1.35e-07,
    'mn5': 6,
    'wn5': 7.5e-07,
    'ln5': 9e-08,
    'cap': 8e-13,
    'res': 9500.0
    }
    parameters_2 = {
    'mp1': 15,
    'wp1': 1.25e-06,
    'lp1': 9e-08,
    'mn1': 10,
    'wn1': 1e-06,
    'ln1': 9e-08,
    'mp3': 12,
    'wp3': 1.25e-06,
    'lp3': 9e-08,
    'mn3': 15,
    'wn3': 1.5e-06,
    'ln3': 9e-08,
    'mn4': 11,
    'wn4': 1.5e-06,
    'ln4': 9e-08,
    'mn5': 11,
    'wn5': 1e-06,
    'ln5': 9e-08,
    'cap': 4.5e-12,
    'res': 4800.0
    }
    parameters_3 = {
    'mp1': 7,
    'wp1': 1.75e-06,
    'lp1': 1.35e-07,
    'mn1': 20,
    'wn1': 2.25e-06,
    'ln1': 1.35e-07,
    'mp3': 22,
    'wp3': 2.25e-06,
    'lp3': 1.35e-07,
    'mn3': 21,
    'wn3': 1e-06,
    'ln3': 9e-08,
    'mn4': 8,
    'wn4': 7.5e-07,
    'ln4': 9e-08,
    'mn5': 17,
    'wn5': 2.5e-07,
    'ln5': 4.5e-08,
    'cap': 8e-13,
    'res': 3900.0
    }
    parameters_4 = {
    'mp1': 8,
    'wp1': 1.5e-06,
    'lp1': 1.35e-07,
    'mn1': 15,
    'wn1': 1.75e-06,
    'ln1': 1.35e-07,
    'mp3': 18,
    'wp3': 2e-06,
    'lp3': 1.35e-07,
    'mn3': 15,
    'wn3': 1.25e-06,
    'ln3': 9e-08,
    'mn4': 13,
    'wn4': 1.5e-06,
    'ln4': 9e-08,
    'mn5': 17,
    'wn5': 5e-07,
    'ln5': 4.5e-08,
    'cap': 3.5e-12,
    'res': 6000.0
    }


    process = "TT"
    temp_pvt = 27
    vdd = 1.2
    dut =DUT(yaml_path)
    new_netlist_path = dut.create_new_netlist(parameters_1, process, temp_pvt, vdd)
    info = dut.simulate(new_netlist_path)
    print(f"New netlist created at: {new_netlist_path}")
    print("info:", info)
    print("trf:", dut.trf)
    print("period:", dut.period)
    print("VDD:", dut.VDD)

    spec_dict = dut.measure_metrics()
    print("gain is: ",spec_dict["gain"])
    print("ugbw is: ",spec_dict["ugbw"])
    print("slew rate is: ",spec_dict['slewRate'])
    print("phm is: ",spec_dict['phm'])
    print("area is: ",spec_dict['area']) 
    print("current is: ",spec_dict['current']) 
    print("total noise is: ",spec_dict['noise'])       
    # area_estimator = BPTM45nmAreaEstimator(self.circuit_params, self.circuit_multipliers)
    # spec_dict['area'] = area_estimator.find_area()
    # spec_dict['current'] = self.current
    # spec_dict['gain'] = self.find_dc_gain(self.vout_complex)
    # spec_dict['noise'] = self.noise
    # spec_dict['phm'] = self.find_phm(self.freq, self.vout_complex)
    # spec_dict['slewRate'] = self.find_slew_rate(self.time, self.vout_tran, threshold_low=0.1, threshold_high=0.9, time_unit='us')
    # spec_dict['ugbw'] = self.find_ugbw(self.freq, self.vout_complex)  
    

    