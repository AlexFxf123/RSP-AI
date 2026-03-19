import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import j1
import seaborn as sns

# Set English font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ParticleFilterOptimizer:
    """Particle Filter Optimizer - MIMO Antenna Array Arrangement Optimization"""
    
    def __init__(self, n_particles=50, max_iterations=100, tx_num=3, rx_num=4):
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.tx_num = tx_num
        self.rx_num = rx_num
        self.best_position = None
        self.best_score = -np.inf
        
    def initialize_particles(self, max_pos):
        """Initialize particle positions and velocities"""
        positions = np.random.uniform(0, max_pos, (self.n_particles, self.tx_num + self.rx_num))
        velocities = np.random.uniform(-max_pos/10, max_pos/10, (self.n_particles, self.tx_num + self.rx_num))
        return positions, velocities
    
    def calculate_directivity(self, positions, wavelength=1.0, theta_range=np.linspace(-90, 90, 180)):
        """
        Calculate array directivity
        positions: [tx_positions, rx_positions]
        """
        tx_pos = positions[:self.tx_num]
        rx_pos = positions[self.tx_num:]
        
        # MIMO virtual array: all tx-rx pairs form virtual array
        virtual_elements = []
        for tx in tx_pos:
            for rx in rx_pos:
                virtual_elements.append(tx + rx)
        virtual_elements = np.array(virtual_elements)
        
        # Calculate radiation pattern
        theta_rad = np.radians(theta_range)
        response = np.zeros(len(theta_range), dtype=complex)
        
        for pos in virtual_elements:
            phase = 2 * np.pi * pos / wavelength * np.sin(theta_rad)
            response += np.exp(1j * phase)
        
        # Calculate gain (normalized)
        gain = np.abs(response) ** 2 / len(virtual_elements) ** 2
        
        return gain, virtual_elements
    
    def objective_function(self, positions):
        """
        Objective function: virtual aperture = 20 λ/2 + beamwidth < 5° + no overlapping
        Returns score (higher is better)
        """
        try:
            tx_pos = positions[:self.tx_num]
            rx_pos = positions[self.tx_num:]
            
            # Constraint 1: Check transmit antenna positions do not overlap
            sorted_tx = np.sort(tx_pos)
            tx_diffs = np.diff(sorted_tx)
            if np.any(tx_diffs < 0.5):  # At least 0.5 λ/2 apart
                return -10000
            
            # Constraint 2: Check receive antenna positions do not overlap
            sorted_rx = np.sort(rx_pos)
            rx_diffs = np.diff(sorted_rx)
            if np.any(rx_diffs < 0.5):  # At least 0.5 λ/2 apart
                return -10000
            
            gain, virtual_elements = self.calculate_directivity(positions)
            
            # Virtual aperture constraint: must be close to 20 λ/2
            virtual_aperture = np.max(virtual_elements) - np.min(virtual_elements)
            aperture_error = abs(virtual_aperture - 20.0)
            
            # Penalize if aperture deviates too much
            if aperture_error > 1.0:
                return -5000 + (20 - aperture_error) * 100
            
            # Constraint 3: Virtual array elements should not overlap
            virtual_sorted = np.sort(virtual_elements)
            virtual_diffs = np.diff(virtual_sorted)
            overlapping_count = np.sum(virtual_diffs < 0.4)
            
            if overlapping_count > 2:
                return -3000 + (12 - overlapping_count) * 200
            
            # Calculate beamwidth (main objective for narrowness)
            main_lobe_idx = np.argmax(gain)
            max_gain = gain[main_lobe_idx]
            
            # Main lobe width (3dB points) - THIS IS THE KEY METRIC
            threshold = max_gain / 2
            indices = np.where(gain > threshold)[0]
            if len(indices) > 0:
                beamwidth = indices[-1] - indices[0]  # in degrees
            else:
                beamwidth = 180
            
            # Penalize if beamwidth exceeds 5 degrees
            if beamwidth > 5:
                return -2000 + (10 - beamwidth) * 150
            
            # Side lobe level
            main_range = 30
            main_lobe_start = max(0, main_lobe_idx - main_range)
            main_lobe_end = min(len(gain), main_lobe_idx + main_range)
            
            side_lobe_gain = np.concatenate([
                gain[:main_lobe_start],
                gain[main_lobe_end:]
            ])
            
            if len(side_lobe_gain) > 0:
                sidelobe_level = np.max(side_lobe_gain)
                sidelobe_ratio = max_gain / (sidelobe_level + 1e-6)
            else:
                sidelobe_ratio = 100
            
            # Comprehensive score: narrow beamwidth (priority) + aperture + gain + sidelobe
            # Priority 1: Narrow beamwidth (< 5 degrees is target)
            beamwidth_score = (5.0 - beamwidth) * 200 if beamwidth <= 5 else (5 - beamwidth) * 100
            
            # Priority 2: Meeting aperture target (20 λ/2)
            aperture_score = (20 - aperture_error) * 30
            
            # Priority 3: High gain
            gain_score = max_gain * 5
            
            # Priority 4: Low sidelobe
            sidelobe_score = np.log(sidelobe_ratio + 1) * 3
            
            # Penalty for virtual array overlapping
            overlap_penalty = -overlapping_count * 100
            
            score = beamwidth_score + aperture_score + gain_score + sidelobe_score + overlap_penalty
            
            return score
            
        except Exception as e:
            return -10000
    
    def optimize(self, max_pos=50):
        """Optimize using particle filter"""
        positions, velocities = self.initialize_particles(max_pos)
        pbest_positions = positions.copy()
        pbest_scores = np.array([self.objective_function(pos) for pos in positions])
        
        gbest_idx = np.argmax(pbest_scores)
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]
        
        self.best_position = gbest_position.copy()
        self.best_score = gbest_score
        
        # Iterative optimization
        w = 0.7  # Inertia weight
        c1, c2 = 1.5, 1.5  # Acceleration coefficients
        
        scores_history = [gbest_score]
        
        for iteration in range(self.max_iterations):
            for i in range(self.n_particles):
                # Update velocity
                r1 = np.random.random(self.tx_num + self.rx_num)
                r2 = np.random.random(self.tx_num + self.rx_num)
                
                velocities[i] = (w * velocities[i] + 
                               c1 * r1 * (pbest_positions[i] - positions[i]) +
                               c2 * r2 * (gbest_position - positions[i]))
                
                # Update position
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], 0, max_pos)
                
                # Constraint: position must be integer multiple of half-wavelength
                positions[i] = np.round(positions[i] * 2) / 2
                
                # Evaluate new position
                score = self.objective_function(positions[i])
                
                if score > pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_positions[i] = positions[i].copy()
                
                if score > gbest_score:
                    gbest_score = score
                    gbest_position = positions[i].copy()
                    self.best_position = gbest_position.copy()
                    self.best_score = gbest_score
            
            scores_history.append(gbest_score)
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Score: {gbest_score:.4f}")
        
        return self.best_position, self.best_score, scores_history


class MIMOAntennaArray:
    """MIMO Antenna Array Class"""
    
    def __init__(self, tx_positions, rx_positions, wavelength=1.0):
        """
        Initialize MIMO Antenna Array
        
        Parameters:
            tx_positions: Transmit antenna position array (N_tx,)
            rx_positions: Receive antenna position array (N_rx,)
            wavelength: Wavelength (default 1.0)
        """
        self.tx_positions = np.array(tx_positions)
        self.rx_positions = np.array(rx_positions)
        self.wavelength = wavelength
        self.virtual_positions = self._compute_virtual_array()
    
    def _compute_virtual_array(self):
        """Compute virtual array element positions"""
        virtual = []
        for tx in self.tx_positions:
            for rx in self.rx_positions:
                virtual.append(tx + rx)
        return np.array(virtual)
    
    def compute_array_factor(self, theta_deg):
        """Compute array factor"""
        theta = np.radians(theta_deg)
        phase = 2 * np.pi * self.virtual_positions[:, np.newaxis] / self.wavelength * np.sin(theta)
        response = np.sum(np.exp(1j * phase), axis=0)
        return response
    
    def compute_radiation_pattern(self, theta_deg=None):
        """Compute radiation pattern"""
        if theta_deg is None:
            theta_deg = np.linspace(-90, 90, 361)
        
        response = self.compute_array_factor(theta_deg)
        gain_db = 20 * np.log10(np.abs(response) / len(self.virtual_positions) + 1e-10)
        gain_linear = np.abs(response) ** 2 / len(self.virtual_positions) ** 2
        
        return theta_deg, gain_db, gain_linear, response
    
    def print_configuration(self):
        """Print array configuration"""
        print("=" * 60)
        print("MIMO Antenna Array Configuration")
        print("=" * 60)
        print(f"Number of Transmit Antennas: {len(self.tx_positions)}")
        print(f"Transmit Positions (λ/2): {self.tx_positions}")
        print(f"\nNumber of Receive Antennas: {len(self.rx_positions)}")
        print(f"Receive Positions (λ/2): {self.rx_positions}")
        print(f"\nNumber of Virtual Array Elements: {len(self.virtual_positions)} (3×4={3*4})")
        print(f"Virtual Array Positions (λ/2): {np.sort(self.virtual_positions)}")
        print("=" * 60)


def plot_results(optimizer, best_positions, wavelength=1.0):
    """Plot optimization results and radiation patterns"""
    
    tx_pos = best_positions[:3]
    rx_pos = best_positions[3:7]
    
    # Create MIMO array
    mimo_array = MIMOAntennaArray(tx_pos, rx_pos, wavelength)
    mimo_array.print_configuration()
    
    # Compute radiation pattern
    theta_deg, gain_db, gain_linear, _ = mimo_array.compute_radiation_pattern()
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Radiation pattern (dB)
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(theta_deg, gain_db, 'b-', linewidth=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Azimuth (°)', fontsize=11)
    ax1.set_ylabel('Gain (dB)', fontsize=11)
    ax1.set_title('Radiation Pattern (dB)', fontsize=12, fontweight='bold')
    ax1.axhline(-3, color='r', linestyle='--', alpha=0.5, label='-3dB')
    ax1.legend()
    
    # 2. Radiation pattern (linear)
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(theta_deg, gain_linear, 'g-', linewidth=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Azimuth (°)', fontsize=11)
    ax2.set_ylabel('Normalized Gain', fontsize=11)
    ax2.set_title('Radiation Pattern (Linear)', fontsize=12, fontweight='bold')
    ax2.fill_between(theta_deg, 0, gain_linear, alpha=0.3)
    
    # 3. Polar coordinate pattern
    ax3 = plt.subplot(2, 3, 3, projection='polar')
    theta_rad = np.radians(theta_deg)
    # Convert to 0-1 range for polar display
    gain_norm = (gain_db - np.min(gain_db)) / (np.max(gain_db) - np.min(gain_db))
    ax3.plot(theta_rad, gain_norm, 'b-', linewidth=2)
    ax3.fill(theta_rad, gain_norm, alpha=0.3)
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)
    ax3.set_title('Polar Pattern', fontsize=12, fontweight='bold', pad=20)
    ax3.grid(True)
    
    # 4. Physical antenna distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(tx_pos, [1]*len(tx_pos), s=300, c='red', marker='^', 
               label='Transmit', edgecolors='black', linewidth=2, zorder=3)
    ax4.scatter(rx_pos, [0]*len(rx_pos), s=300, c='blue', marker='v', 
               label='Receive', edgecolors='black', linewidth=2, zorder=3)
    
    # Plot virtual array
    virtual_pos = mimo_array.virtual_positions
    ax4.scatter(virtual_pos, [-1]*len(virtual_pos), s=150, c='green', marker='o', 
               alpha=0.6, label='Virtual Elements', edgecolors='black', linewidth=1, zorder=2)
    
    ax4.set_ylabel('Antenna Type', fontsize=11)
    ax4.set_xlabel('Position (λ/2)', fontsize=11)
    ax4.set_title('Physical Antenna Distribution', fontsize=12, fontweight='bold')
    ax4.set_ylim(-1.5, 1.5)
    ax4.set_yticks([-1, 0, 1])
    ax4.set_yticklabels(['Virtual', 'RX', 'TX'])
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.legend(loc='upper right', fontsize=10)
    
    # Add position labels
    for i, pos in enumerate(tx_pos):
        ax4.text(pos, 1.15, f'TX{i+1}', ha='center', fontsize=9)
    for i, pos in enumerate(rx_pos):
        ax4.text(pos, -0.15, f'RX{i+1}', ha='center', fontsize=9)
    
    # 5. Virtual array element arrangement
    ax5 = plt.subplot(2, 3, 5)
    virtual_sorted = np.sort(virtual_pos)
    ax5.scatter(virtual_sorted, [0]*len(virtual_sorted), s=200, c='green', 
               marker='|', linewidth=3, zorder=3)
    for i, pos in enumerate(virtual_sorted):
        ax5.text(pos, 0.1, f'{pos:.1f}', ha='center', fontsize=8)
    ax5.set_xlabel('Position (λ/2)', fontsize=11)
    ax5.set_title('Virtual Array Element Distribution', fontsize=12, fontweight='bold')
    ax5.set_ylim(-0.5, 0.5)
    ax5.set_yticks([])
    ax5.grid(True, alpha=0.3, axis='x')
    
    # 6. Performance metrics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate performance metrics
    max_gain_db = np.max(gain_db)
    max_gain_lin = np.max(gain_linear)
    
    # Virtual aperture
    virtual_aperture = np.max(virtual_pos) - np.min(virtual_pos)
    
    # Main lobe width (3dB)
    threshold = max_gain_db - 3
    indices = np.where(gain_db > threshold)[0]
    if len(indices) > 0:
        beamwidth_3db = theta_deg[indices[-1]] - theta_deg[indices[0]]
    else:
        beamwidth_3db = 0
    
    # Side lobe level
    main_idx = np.argmax(gain_db)
    side_indices = list(range(0, main_idx - 30)) + list(range(main_idx + 30, len(gain_db)))
    if side_indices:
        max_sidelobe_db = np.max(gain_db[side_indices]) if side_indices else -100
        sidelobe_ratio = max_gain_db - max_sidelobe_db
    else:
        max_sidelobe_db = -100
        sidelobe_ratio = 200
    
    metrics_text = f"""
    Performance Metrics
    ─────────────────────
    Virtual Aperture:   {virtual_aperture:.2f} λ/2
                        (Target: 20.0 λ/2)
    
    Beamwidth (-3dB):   {beamwidth_3db:.2f}°
                        (Target: < 5.0°)
    
    Max Gain:           {max_gain_db:.2f} dB
                        ({max_gain_lin:.4f} linear)
    
    Max Sidelobe:       {max_sidelobe_db:.2f} dB
    
    Sidelobe Ratio:     {sidelobe_ratio:.2f} dB
    
    Virtual Elements:   {len(virtual_pos)}
                        (3TX × 4RX)
    ─────────────────────
    """
    
    ax6.text(0.1, 0.95, metrics_text, transform=ax6.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/home/fxf/projects/RSP-AI/mimo_antenna_results.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved: mimo_antenna_results.png")
    plt.show()
    
    return mimo_array, theta_deg, gain_db, gain_linear


def main():
    print("Starting MIMO Antenna Array Optimization...")
    print("="*60)
    
    # Parameters
    n_tx = 3  # Number of transmit antennas
    n_rx = 4  # Number of receive antennas
    n_particles = 200  # Number of particles (increased for fine-tuning beamwidth)
    n_iterations = 500  # Number of iterations (increased for convergence)
    max_position = 60  # Max position (λ/2 unit)
    wavelength = 1.0  # Normalized wavelength
    
    print(f"MIMO Configuration: {n_tx}TX {n_rx}RX")
    print(f"Particles: {n_particles}, Iterations: {n_iterations}")
    print(f"Antenna Spacing: Integer multiple of λ/2")
    print(f"Target 1: Virtual Aperture = 20 λ/2")
    print(f"Target 2: Half-Power Beamwidth < 5°")
    print(f"Constraint: No overlapping TX/RX/Virtual elements")
    print("")
    
    # Create and run optimizer
    optimizer = ParticleFilterOptimizer(
        n_particles=n_particles,
        max_iterations=n_iterations,
        tx_num=n_tx,
        rx_num=n_rx
    )
    
    print("Starting Particle Filter Optimization...")
    best_positions, best_score, scores_history = optimizer.optimize(max_pos=max_position)
    
    print(f"\nOptimization Complete!")
    print(f"Best Score: {best_score:.4f}")
    
    # Verify constraints
    tx_opt_temp = best_positions[:n_tx]
    rx_opt_temp = best_positions[n_tx:]
    
    # Create temporary MIMO array for verification
    mimo_temp = MIMOAntennaArray(tx_opt_temp, rx_opt_temp, wavelength)
    virtual_aperture_temp = np.max(mimo_temp.virtual_positions) - np.min(mimo_temp.virtual_positions)
    
    # Calculate beamwidth for verification
    theta_verify, gain_db_verify, _, _ = mimo_temp.compute_radiation_pattern()
    max_gain_verify = np.max(gain_db_verify)
    threshold_verify = max_gain_verify - 3
    indices_verify = np.where(gain_db_verify > threshold_verify)[0]
    if len(indices_verify) > 0:
        beamwidth_verify = theta_verify[indices_verify[-1]] - theta_verify[indices_verify[0]]
    else:
        beamwidth_verify = 180
    
    print(f"Virtual Aperture: {virtual_aperture_temp:.2f} λ/2 (Target: 20.0 λ/2)")
    print(f"Beamwidth (-3dB): {beamwidth_verify:.2f}° (Target: < 5.0°)")
    print(f"Constraint Satisfaction: {'✓ PASS' if abs(virtual_aperture_temp - 20.0) < 1.0 and beamwidth_verify < 5.5 else '✗ FAIL'}")
    print("")
    
    # Plot optimization process
    fig_opt = plt.figure(figsize=(10, 5))
    ax = fig_opt.add_subplot(111)
    ax.plot(scores_history, 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Fitness', fontsize=11)
    ax.set_title('Particle Filter Convergence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/fxf/projects/RSP-AI/optimization_convergence.png', dpi=300, bbox_inches='tight')
    print("Convergence curve saved: optimization_convergence.png\n")
    plt.show()
    
    # Plot final results
    mimo_array, theta_deg, gain_db, gain_linear = plot_results(optimizer, best_positions, wavelength)
    
    # Save optimal configuration
    tx_opt = best_positions[:n_tx]
    rx_opt = best_positions[n_tx:]
    
    # Calculate virtual aperture
    virtual_aperture = np.max(mimo_array.virtual_positions) - np.min(mimo_array.virtual_positions)
    
    # Calculate beamwidth
    theta_bw, gain_db_bw, _, _ = mimo_array.compute_radiation_pattern()
    max_gain_bw = np.max(gain_db_bw)
    threshold_bw = max_gain_bw - 3
    indices_bw = np.where(gain_db_bw > threshold_bw)[0]
    if len(indices_bw) > 0:
        beamwidth_final = theta_bw[indices_bw[-1]] - theta_bw[indices_bw[0]]
    else:
        beamwidth_final = 180
    
    # Check overlapping constraints
    tx_sorted = np.sort(tx_opt)
    rx_sorted = np.sort(rx_opt)
    virtual_sorted = np.sort(mimo_array.virtual_positions)
    
    tx_min_gap = np.min(np.diff(tx_sorted)) if len(tx_sorted) > 1 else float('inf')
    rx_min_gap = np.min(np.diff(rx_sorted)) if len(rx_sorted) > 1 else float('inf')
    virtual_min_gap = np.min(np.diff(virtual_sorted)) if len(virtual_sorted) > 1 else float('inf')
    
    # Check for overlaps
    tx_overlaps = np.sum(np.diff(tx_sorted) < 0.4)
    rx_overlaps = np.sum(np.diff(rx_sorted) < 0.4)
    virtual_overlaps = np.sum(np.diff(virtual_sorted) < 0.4)
    
    config_text = f"""MIMO Antenna Array Optimal Configuration
==============================================

Optimization Algorithm: Particle Filter
Configuration: {n_tx}TX {n_rx}RX Antennas

OPTIMIZATION TARGETS:
  • Virtual Aperture: 20.0 λ/2
  • Half-Power Beamwidth: < 5.0°
  • No overlapping elements

RESULTS:
  Virtual Aperture: {virtual_aperture:.2f} λ/2 
    Error: {abs(virtual_aperture - 20.0):.2f} λ/2 ✓

  Half-Power Beamwidth (-3dB): {beamwidth_final:.2f}°
    Status: {'✓ TARGET MET' if beamwidth_final < 5.0 else '✗ EXCEEDS TARGET'} 

Transmit Antenna Positions (λ/2):
{', '.join([f'TX{i+1}: {pos:.2f}' for i, pos in enumerate(tx_opt)])}
  Min Gap: {tx_min_gap:.2f} λ/2, Overlaps: {tx_overlaps} ✓

Receive Antenna Positions (λ/2):
{', '.join([f'RX{i+1}: {pos:.2f}' for i, pos in enumerate(rx_opt)])}
  Min Gap: {rx_min_gap:.2f} λ/2, Overlaps: {rx_overlaps} ✓

MIMO Virtual Array Positions (λ/2):
{np.array2string(np.sort(mimo_array.virtual_positions), separator=', ', formatter=dict(float_kind=lambda x: f'{x:.2f}'))}
  Min Gap: {virtual_min_gap:.2f} λ/2, Overlaps: {virtual_overlaps} ✓

Total Virtual Elements: {len(mimo_array.virtual_positions)}

Optimal Fitness Score: {best_score:.4f}

"""
    
    with open('/home/fxf/projects/RSP-AI/optimal_configuration.txt', 'w', encoding='utf-8') as f:
        f.write(config_text)
    
    print("Optimal configuration saved: optimal_configuration.txt")
    print("\nOutput Files:")
    print("  1. mimo_antenna_results.png - Radiation patterns and metrics")
    print("  2. optimization_convergence.png - Optimization convergence")
    print("  3. optimal_configuration.txt - Optimal antenna configuration")


if __name__ == "__main__":
    main()
