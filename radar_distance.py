import numpy as np
import matplotlib.pyplot as plt
import math

class FMCWRadar:
    """FMCW Radar Detection Range Calculator (Latest Formula)"""
    
    def __init__(self):
        """Initialize radar parameters"""
        # Radar system parameters
        self.pt_dbm = 12.0  # Transmit power, dBm
        self.gain_tx_db = 20.0  # Transmit antenna gain, dB
        self.gain_rx_db = 20.0  # Receive antenna gain, dB
        self.freq_hz = 77.0e9  # Frequency, Hz
        self.t_mod_s = 20.0e-6  # Modulation time, s
        self.gs_db = 30.0  # Signal processing gain, dB
        
        # Environmental parameters
        self.temp_k = 300.0  # Operating temperature, K
        self.noise_figure_db = 12.0  # Receiver noise figure, dB
        self.snr_min_db = 12.0  # Minimum detection SNR, dB
        
        # Target parameters
        self.rcs_m2 = 0.01  # Target RCS, m²
        
        # Physical constants
        self.c = 3.0e8  # Speed of light, m/s
        self.k = 1.380649e-23  # Boltzmann constant, J/K
        self.pi = math.pi
    
    def dbm_to_watt(self, dbm):
        """Convert dBm to watts"""
        return 1e-3 * 10**(dbm/10.0)
    
    def db_to_linear(self, db):
        """Convert dB to linear value"""
        return 10**(db/10.0)
    
    def calculate_wavelength(self):
        """Calculate wavelength"""
        return self.c / self.freq_hz
    
    def calculate_range(self, rcs=None):
        """
        Calculate detection range based on the latest radar equation
        
        Latest formula: R_max^4 = (P_t * G_t * G_r * λ^2 * σ * G_s * t_mod) / ((4π)^3 * k * T * F * (S/N)_{min})
        
        Parameters:
        rcs: Target RCS, m². If None, use instance's rcs_m2
        
        Returns:
        Detection range, meters
        """
        if rcs is None:
            rcs = self.rcs_m2
        
        # Parameter conversion
        pt_w = self.dbm_to_watt(self.pt_dbm)  # Transmit power, W
        g_t = self.db_to_linear(self.gain_tx_db)  # Transmit antenna gain (linear)
        g_r = self.db_to_linear(self.gain_rx_db)  # Receive antenna gain (linear)
        wavelength = self.calculate_wavelength()  # Wavelength, m
        t = self.temp_k  # Temperature, K
        f_linear = self.db_to_linear(self.noise_figure_db)  # Noise figure (linear)
        snr_min_linear = self.db_to_linear(self.snr_min_db)  # Minimum SNR (linear)
        gs_linear = self.db_to_linear(self.gs_db)  # Signal processing gain (linear)
        t_mod = self.t_mod_s  # Modulation time, s
        
        # Latest radar equation: R^4 = (Pt * G_t * G_r * λ^2 * σ * G_s * t_mod) / ((4π)^3 * k * T * F * SNR_min)
        numerator = pt_w * g_t * g_r * wavelength**2 * rcs * gs_linear * t_mod
        denominator = (4 * self.pi)**3 * self.k * t * f_linear * snr_min_linear
        
        if denominator <= 0:
            return 0.0
        
        r4 = numerator / denominator
        if r4 <= 0:
            return 0.0
            
        range_m = r4**0.25  # R = (R^4)^(1/4)
        return range_m
    
    def calculate_max_unambiguous_range(self):
        """Calculate maximum unambiguous range"""
        return self.c * self.t_mod_s / 2
    
    def print_parameters(self):
        """Print radar parameters"""
        pt_w = self.dbm_to_watt(self.pt_dbm)
        wavelength = self.calculate_wavelength()
        nf_linear = self.db_to_linear(self.noise_figure_db)
        snr_min_linear = self.db_to_linear(self.snr_min_db)
        gs_linear = self.db_to_linear(self.gs_db)
        g_t_linear = self.db_to_linear(self.gain_tx_db)
        g_r_linear = self.db_to_linear(self.gain_rx_db)
        range_unamb = self.calculate_max_unambiguous_range()
        range_m = self.calculate_range()
        
        print("=" * 90)
        print("FMCW Radar Parameters Configuration Table (Latest Formula)")
        print("=" * 90)
        print(f"{'Parameter':<25} {'Value':<20} {'Unit':<15} {'Description'}")
        print("-" * 90)
        print(f"{'Target RCS (σ)':<25} {self.rcs_m2:<20.4f} {'m²':<15} Radar Cross Section")
        print(f"{'Transmit Power (P_t)':<25} {self.pt_dbm:<20.1f} {'dBm':<15} Linear value: {pt_w:.3e} W")
        print(f"{'Tx Antenna Gain (G_t)':<25} {self.gain_tx_db:<20.1f} {'dB':<15} Linear value: {g_t_linear:.3f}")
        print(f"{'Rx Antenna Gain (G_r)':<25} {self.gain_rx_db:<20.1f} {'dB':<15} Linear value: {g_r_linear:.3f}")
        print(f"{'Signal Processing Gain (G_s)':<25} {self.gs_db:<20.1f} {'dB':<15} Linear value: {gs_linear:.3f}")
        print(f"{'Frequency (f)':<25} {self.freq_hz/1e9:<20.1f} {'GHz':<15} Wavelength: {wavelength*100:.3f} cm")
        print(f"{'Modulation Time (T_mod)':<25} {self.t_mod_s*1e6:<20.1f} {'μs':<15} 20 μs")
        print(f"{'Temperature (T)':<25} {self.temp_k:<20.0f} {'K':<15} Operating temperature")
        print(f"{'Noise Figure (F)':<25} {self.noise_figure_db:<20.1f} {'dB':<15} Linear value: {nf_linear:.3f}")
        print(f"{'Min SNR (S/N_min)':<25} {self.snr_min_db:<20.1f} {'dB':<15} Linear value: {snr_min_linear:.3f}")
        print(f"{'Max Unambiguous Range':<25} {range_unamb:<20.1f} {'m':<15} R_unamb = c×T_mod/2")
        print("-" * 90)
        print(f"{'Estimated Detection Range':<25} {range_m:<20.2f} {'m':<15} About {range_m/1000:.2f} km")
        print("=" * 90)
        print()


def plot_range_vs_rcs(radar, rcs_min=0.001, rcs_max=0.1, num_points=100):
    """
    Plot detection range vs RCS
    """
    # Generate RCS range
    rcs_values = np.linspace(rcs_min, rcs_max, num_points)
    range_values = np.zeros_like(rcs_values)
    
    # Calculate detection range for each RCS
    for i, rcs in enumerate(rcs_values):
        range_values[i] = radar.calculate_range(rcs)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Detection range vs RCS
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(rcs_values, range_values, 'b-', linewidth=2)
    ax1.set_xlabel('Target RCS (m²)', fontsize=12)
    ax1.set_ylabel('Detection Range (m)', fontsize=12)
    ax1.set_title('Detection Range vs Target RCS', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # Mark current configuration point
    current_range = radar.calculate_range(radar.rcs_m2)
    ax1.plot(radar.rcs_m2, current_range, 'ro', markersize=10, 
             label=f'RCS={radar.rcs_m2} m²\nR={current_range:.1f} m')
    ax1.legend(fontsize=10)
    
    # Detection range vs RCS (log-log scale)
    ax2 = plt.subplot(2, 2, 2)
    ax2.loglog(rcs_values, range_values, 'r-', linewidth=2)
    ax2.set_xlabel('Target RCS (m²)', fontsize=12)
    ax2.set_ylabel('Detection Range (m)', fontsize=12)
    ax2.set_title('Detection Range vs Target RCS (Log-Log Scale)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    # Range vs modulation time
    ax3 = plt.subplot(2, 2, 3)
    t_mod_values = np.linspace(1, 100, 50)  # 1 to 100 μs
    range_vs_t_mod = np.zeros_like(t_mod_values)
    
    for i, t_mod in enumerate(t_mod_values):
        radar_temp = FMCWRadar()
        radar_temp.t_mod_s = t_mod * 1e-6
        range_vs_t_mod[i] = radar_temp.calculate_range()
    
    ax3.plot(t_mod_values, range_vs_t_mod, 'g-', linewidth=2)
    ax3.set_xlabel('Modulation Time T_mod (μs)', fontsize=12)
    ax3.set_ylabel('Detection Range (m)', fontsize=12)
    ax3.set_title('Detection Range vs Modulation Time', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=10)
    current_t_mod = radar.t_mod_s * 1e6
    ax3.plot(current_t_mod, current_range, 'ro', markersize=10)
    
    # Range vs antenna gain (assuming G_t = G_r)
    ax4 = plt.subplot(2, 2, 4)
    gain_values = np.linspace(0, 30, 50)  # 0 to 30 dB
    range_vs_gain = np.zeros_like(gain_values)
    
    for i, gain in enumerate(gain_values):
        radar_temp = FMCWRadar()
        radar_temp.gain_tx_db = gain
        radar_temp.gain_rx_db = gain
        range_vs_gain[i] = radar_temp.calculate_range()
    
    ax4.plot(gain_values, range_vs_gain, 'm-', linewidth=2, label='Detection Range')
    ax4.set_xlabel('Antenna Gain (dB, G_t = G_r)', fontsize=12)
    ax4.set_ylabel('Detection Range (m)', fontsize=12)
    ax4.set_title('Detection Range vs Antenna Gain', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='both', which='major', labelsize=10)
    ax4.plot(radar.gain_tx_db, current_range, 'ro', markersize=10)
    
    plt.tight_layout()
    plt.show()
    
    return rcs_values, range_values


def plot_range_vs_processing_gain_and_power(radar):
    """Plot detection range vs signal processing gain and transmit power"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Range vs signal processing gain
    gs_values = np.linspace(0, 50, 51)  # 0 to 50 dB
    range_vs_gs = np.zeros_like(gs_values)
    
    for i, gs in enumerate(gs_values):
        radar_temp = FMCWRadar()
        radar_temp.gs_db = gs
        range_vs_gs[i] = radar_temp.calculate_range()
    
    ax1.plot(gs_values, range_vs_gs, 'b-', linewidth=2)
    ax1.set_xlabel('Signal Processing Gain G_s (dB)', fontsize=12)
    ax1.set_ylabel('Detection Range (m)', fontsize=12)
    ax1.set_title('Detection Range vs Signal Processing Gain', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=radar.gs_db, color='r', linestyle='--', alpha=0.7, 
                label=f'Current G_s: {radar.gs_db} dB')
    ax1.legend(fontsize=10)
    
    # Range vs transmit power
    power_values = np.linspace(-20, 20, 41)  # -20 to 20 dBm
    range_vs_power = np.zeros_like(power_values)
    
    for i, power in enumerate(power_values):
        radar_temp = FMCWRadar()
        radar_temp.pt_dbm = power
        range_vs_power[i] = radar_temp.calculate_range()
    
    ax2.plot(power_values, range_vs_power, 'g-', linewidth=2)
    ax2.set_xlabel('Transmit Power (dBm)', fontsize=12)
    ax2.set_ylabel('Detection Range (m)', fontsize=12)
    ax2.set_title('Detection Range vs Transmit Power', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=radar.pt_dbm, color='r', linestyle='--', alpha=0.7, 
                label=f'Current Power: {radar.pt_dbm} dBm')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_range_vs_frequency_and_snr(radar):
    """Plot detection range vs frequency and SNR"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Range vs frequency
    freq_values = np.linspace(24, 120, 50)  # 24 to 120 GHz
    range_vs_freq = np.zeros_like(freq_values)
    
    for i, freq in enumerate(freq_values):
        radar_temp = FMCWRadar()
        radar_temp.freq_hz = freq * 1e9
        range_vs_freq[i] = radar_temp.calculate_range()
    
    ax1.plot(freq_values, range_vs_freq, 'b-', linewidth=2)
    ax1.set_xlabel('Frequency (GHz)', fontsize=12)
    ax1.set_ylabel('Detection Range (m)', fontsize=12)
    ax1.set_title('Detection Range vs Frequency', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=radar.freq_hz/1e9, color='r', linestyle='--', alpha=0.7, 
                label=f'Current freq: {radar.freq_hz/1e9:.1f} GHz')
    ax1.legend(fontsize=10)
    
    # Range vs minimum SNR
    snr_values = np.linspace(0, 20, 41)  # 0 to 20 dB
    range_vs_snr = np.zeros_like(snr_values)
    
    for i, snr in enumerate(snr_values):
        radar_temp = FMCWRadar()
        radar_temp.snr_min_db = snr
        range_vs_snr[i] = radar_temp.calculate_range()
    
    ax2.plot(snr_values, range_vs_snr, 'g-', linewidth=2)
    ax2.set_xlabel('Minimum SNR (dB)', fontsize=12)
    ax2.set_ylabel('Detection Range (m)', fontsize=12)
    ax2.set_title('Detection Range vs Minimum SNR', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=radar.snr_min_db, color='r', linestyle='--', alpha=0.7, 
                label=f'Current SNR_min: {radar.snr_min_db} dB')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function"""
    print("FMCW Radar Detection Range Calculator (Latest Formula)")
    print("=" * 60)
    print("Latest Formula: R_max^4 = (P_t * G_t * G_r * λ^2 * σ * G_s * t_mod) / ((4π)^3 * k * T * F * (S/N)_{min})")
    print("=" * 60)
    
    # Create radar instance
    radar = FMCWRadar()
    
    # Print parameters
    radar.print_parameters()
    
    # Plot detection range vs RCS
    print("Generating Detection Range vs RCS plot...")
    plot_range_vs_rcs(radar)
    
    # Plot detection range vs signal processing gain and transmit power
    print("Generating Detection Range vs Signal Processing Gain and Transmit Power plot...")
    plot_range_vs_processing_gain_and_power(radar)
    
    # Plot detection range vs frequency and SNR
    print("Generating Detection Range vs Frequency and SNR plot...")
    plot_range_vs_frequency_and_snr(radar)
    
    # Calculate detection range for different RCS values
    print("\nDetection Range Examples for Different RCS:")
    print("-" * 60)
    rcs_examples = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]  # m²
    
    for rcs in rcs_examples:
        distance = radar.calculate_range(rcs)
        print(f"RCS = {rcs:6.3f} m²: Detection Range = {distance:8.2f} m ({distance/1000:6.2f} km)")
    
    # Parameter sensitivity analysis
    print("\nParameter Sensitivity Analysis:")
    print("-" * 60)
    
    # Test different signal processing gains
    test_gs = [0, 10, 20, 30, 40, 50]  # dB
    print("\nDetection Range for Different Signal Processing Gains:")
    for gs in test_gs:
        radar_temp = FMCWRadar()
        radar_temp.gs_db = gs
        distance = radar_temp.calculate_range()
        print(f"G_s = {gs:2d} dB: Detection Range = {distance:7.2f} m")
    
    # Test different antenna gains
    test_gain = [0, 10, 20, 30]  # dB
    print("\nDetection Range for Different Antenna Gains (G_t = G_r):")
    for gain in test_gain:
        radar_temp = FMCWRadar()
        radar_temp.gain_tx_db = gain
        radar_temp.gain_rx_db = gain
        distance = radar_temp.calculate_range()
        print(f"G_t = G_r = {gain:2d} dB: Detection Range = {distance:7.2f} m")
    
    # Test different modulation times
    test_t_mod = [10, 20, 50, 100]  # μs
    print("\nDetection Range for Different Modulation Times:")
    for t_mod in test_t_mod:
        radar_temp = FMCWRadar()
        radar_temp.t_mod_s = t_mod * 1e-6
        distance = radar_temp.calculate_range()
        print(f"T_mod = {t_mod:3d} μs: Detection Range = {distance:7.2f} m")


if __name__ == "__main__":
    main()
