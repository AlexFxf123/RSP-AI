import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
from scipy.signal import find_peaks

class MIMOAntennaArray:
    def __init__(self, num_tx=3, num_rx=4, virtual_aperture=20, target_hpbw=4.0):
        """
        Initialize MIMO antenna array parameters
        
        Parameters:
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas
        virtual_aperture: Virtual aperture (in half-wavelength units)
        target_hpbw: Target half-power beamwidth (degrees)
        """
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.virtual_aperture = virtual_aperture
        self.target_hpbw = target_hpbw
        self.wavelength = 1.0  # Normalized wavelength
        self.half_wavelength = 0.5  # Half wavelength
        
    def calculate_virtual_array(self, tx_positions, rx_positions):
        """
        Calculate MIMO virtual array positions
        
        Parameters:
        tx_positions: Transmit antenna positions (half-wavelength units)
        rx_positions: Receive antenna positions (half-wavelength units)
        
        Returns:
        virtual_positions: Virtual array positions
        """
        virtual_positions = []
        for tx_pos in tx_positions:
            for rx_pos in rx_positions:
                virtual_pos = tx_pos + rx_pos
                virtual_positions.append(virtual_pos)
        return np.sort(np.array(virtual_positions))
    
    def check_constraints(self, tx_positions, rx_positions):
        """
        Check all constraints
        
        Returns:
        violation_score: Constraint violation penalty score
        """
        violation_score = 0
        
        # 1. Check transmit antennas don't overlap
        if len(set(tx_positions)) != len(tx_positions):
            violation_score += 100
        
        # 2. Check virtual antennas don't overlap
        virtual_positions = self.calculate_virtual_array(tx_positions, rx_positions)
        if len(set(virtual_positions)) != len(virtual_positions):
            violation_score += 100
        
        # 3. Check virtual aperture constraint
        if len(virtual_positions) > 0:
            virtual_aperture_actual = virtual_positions[-1] - virtual_positions[0]
            aperture_error = abs(virtual_aperture_actual - self.virtual_aperture)
            violation_score += aperture_error * 10
        else:
            violation_score += 1000
        
        # 4. Check positions are integer multiples of half wavelength
        for pos in tx_positions:
            if not np.isclose(pos, round(pos)):
                violation_score += 50
        for pos in rx_positions:
            if not np.isclose(pos, round(pos)):
                violation_score += 50
        
        return violation_score
    
    def array_factor(self, virtual_positions, theta_degrees):
        """
        Calculate array pattern
        
        Parameters:
        virtual_positions: Virtual array positions
        theta_degrees: Angle (degrees)
        
        Returns:
        array_factor: Array factor
        """
        theta_rad = np.deg2rad(theta_degrees)
        
        # Convert to wavelength units
        k = 2 * np.pi / self.wavelength
        positions_in_wavelength = virtual_positions * self.half_wavelength
        
        # Calculate array factor
        array_factor = np.zeros_like(theta_degrees, dtype=complex)
        for pos in positions_in_wavelength:
            array_factor += np.exp(1j * k * pos * np.sin(theta_rad))
        
        return np.abs(array_factor)
    
    def calculate_hpbw(self, virtual_positions):
        """
        Calculate half-power beamwidth
        
        Returns:
        hpbw: Half-power beamwidth (degrees)
        mainlobe_angle: Main lobe direction (degrees)
        theta: Angle array
        af_normalized: Normalized array factor
        """
        # Angle range
        theta = np.linspace(-90, 90, 1801)
        
        # Calculate pattern
        af = self.array_factor(virtual_positions, theta)
        af_normalized = af / np.max(af)
        
        # Find main lobe peak
        peak_idx = np.argmax(af_normalized)
        mainlobe_angle = theta[peak_idx]
        
        # Find -3dB points (half-power points)
        half_power = 1 / np.sqrt(2)  # Approximately 0.707
        
        # Search left
        left_idx = peak_idx
        while left_idx > 0 and af_normalized[left_idx] > half_power:
            left_idx -= 1
        
        # Search right
        right_idx = peak_idx
        while right_idx < len(theta) - 1 and af_normalized[right_idx] > half_power:
            right_idx += 1
        
        # Linear interpolation for more accurate -3dB points
        if left_idx > 0 and left_idx < len(theta) - 1:
            theta_left = np.interp(half_power, 
                                  [af_normalized[left_idx], af_normalized[left_idx + 1]],
                                  [theta[left_idx], theta[left_idx + 1]])
        else:
            theta_left = theta[left_idx]
        
        if right_idx > 0 and right_idx < len(theta) - 1:
            theta_right = np.interp(half_power,
                                   [af_normalized[right_idx - 1], af_normalized[right_idx]],
                                   [theta[right_idx - 1], theta[right_idx]])
        else:
            theta_right = theta[right_idx]
        
        hpbw = theta_right - theta_left
        
        return hpbw, mainlobe_angle, theta, af_normalized
    
    def find_mainlobe_region(self, theta, af_db):
        """
        Find main lobe region by locating the first minima on both sides of the main lobe peak
        
        Parameters:
        theta: Angle array (degrees)
        af_db: Array factor in dB
        
        Returns:
        left_boundary: Index of left boundary
        right_boundary: Index of right boundary
        left_min_angle: Angle of left minimum
        right_min_angle: Angle of right minimum
        """
        # Find main lobe peak
        peak_idx = np.argmax(af_db)
        
        # Initialize boundaries
        left_boundary = 0
        right_boundary = len(theta) - 1
        
        # Find first minimum to the left of the peak
        for i in range(peak_idx, 0, -1):
            # Check if we're at the left edge
            if i == 0:
                left_boundary = 0
                break
            
            # Check if current point is a local minimum
            # A point is considered a minimum if it's lower than its neighbors
            if i > 0 and i < len(theta) - 1:
                if af_db[i] < af_db[i-1] and af_db[i] < af_db[i+1]:
                    left_boundary = i
                    break
        
        # Find first minimum to the right of the peak
        for i in range(peak_idx, len(theta)-1):
            # Check if we're at the right edge
            if i == len(theta) - 1:
                right_boundary = len(theta) - 1
                break
            
            # Check if current point is a local minimum
            if i > 0 and i < len(theta) - 1:
                if af_db[i] < af_db[i-1] and af_db[i] < af_db[i+1]:
                    right_boundary = i
                    break
        
        left_min_angle = theta[left_boundary]
        right_min_angle = theta[right_boundary]
        
        return left_boundary, right_boundary, left_min_angle, right_min_angle
    
    def calculate_sll(self, virtual_positions):
        """
        Calculate sidelobe level (SLL) - the highest level outside the main lobe
        
        Returns:
        max_sll: Maximum sidelobe level (dB)
        sll_angle: Angle of maximum sidelobe (degrees)
        sll_value: Maximum sidelobe value (dB)
        theta: Angle array
        af_db: Array factor in dB
        left_min_angle: Angle of left main lobe boundary
        right_min_angle: Angle of right main lobe boundary
        """
        # Angle range
        theta = np.linspace(-90, 90, 1801)
        
        # Calculate pattern
        af = self.array_factor(virtual_positions, theta)
        af_normalized = af / np.max(af)
        af_db = 20 * np.log10(af_normalized + 1e-10)
        
        # Find main lobe region using first minima method
        left_boundary, right_boundary, left_min_angle, right_min_angle = self.find_mainlobe_region(theta, af_db)
        
        # Create main lobe region mask
        mainlobe_region = np.zeros(len(theta), dtype=bool)
        mainlobe_region[left_boundary:right_boundary+1] = True
        
        # Find maximum in sidelobe region (outside main lobe)
        sidelobe_mask = ~mainlobe_region
        
        if np.any(sidelobe_mask):
            # Find maximum in sidelobe region
            max_sll_idx = np.argmax(af_db[sidelobe_mask])
            
            # Convert index back to full array
            full_indices = np.where(sidelobe_mask)[0]
            sll_idx = full_indices[max_sll_idx]
            
            max_sll = af_db[sll_idx]
            sll_angle = theta[sll_idx]
            sll_value = max_sll
        else:
            max_sll = -np.inf
            sll_angle = 0.0
            sll_value = -np.inf
        
        return max_sll, sll_angle, sll_value, theta, af_db, left_min_angle, right_min_angle
    
    def calculate_all_sidelobes(self, virtual_positions, threshold_db=-30, min_separation=5.0):
        """
        Calculate all sidelobes above a threshold
        
        Parameters:
        virtual_positions: Virtual array positions
        threshold_db: Threshold for sidelobe detection (dB)
        min_separation: Minimum angular separation between sidelobes (degrees)
        
        Returns:
        sll_angles: Angles of sidelobes (degrees)
        sll_values: Sidelobe values (dB)
        """
        theta = np.linspace(-90, 90, 1801)
        af = self.array_factor(virtual_positions, theta)
        af_normalized = af / np.max(af)
        af_db = 20 * np.log10(af_normalized + 1e-10)
        
        # Find main lobe region
        left_boundary, right_boundary, left_min_angle, right_min_angle = self.find_mainlobe_region(theta, af_db)
        
        # Create main lobe region mask
        mainlobe_region = np.zeros(len(theta), dtype=bool)
        mainlobe_region[left_boundary:right_boundary+1] = True
        
        # Find all peaks above threshold
        peaks, properties = find_peaks(af_db, height=threshold_db, distance=int(min_separation/0.1))
        
        # Filter peaks outside main lobe region
        sidelobe_peaks = [p for p in peaks if not mainlobe_region[p]]
        
        if len(sidelobe_peaks) > 0:
            sll_angles = theta[sidelobe_peaks]
            sll_values = af_db[sidelobe_peaks]
        else:
            sll_angles = np.array([])
            sll_values = np.array([])
        
        return sll_angles, sll_values

class PSOOptimizer:
    def __init__(self, antenna_array, num_particles=50, max_iter=200):
        """
        Initialize PSO optimizer
        
        Parameters:
        antenna_array: MIMOAntennaArray instance
        num_particles: Number of particles
        max_iter: Maximum iterations
        """
        self.antenna_array = antenna_array
        self.num_particles = num_particles
        self.max_iter = max_iter
        
        # Optimization variable dimension: 3 TX positions + 4 RX positions
        self.dim = antenna_array.num_tx + antenna_array.num_rx
        
        # PSO parameters
        self.w = 0.9  # Inertia weight
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        
        # Position and velocity ranges
        self.pos_min = 0
        self.pos_max = antenna_array.virtual_aperture
        
        # Initialize particles
        self.positions = None
        self.velocities = None
        self.pbest_positions = None
        self.pbest_scores = None
        self.gbest_position = None
        self.gbest_score = float('inf')
        
    def initialize_particles(self):
        """Initialize particle swarm"""
        self.positions = np.random.uniform(self.pos_min, self.pos_max, 
                                          (self.num_particles, self.dim))
        
        # Ensure positions are integers (integer multiples of half-wavelength)
        self.positions = np.round(self.positions)
        
        # Ensure transmit antennas don't overlap
        for i in range(self.num_particles):
            tx_pos = self.positions[i, :self.antenna_array.num_tx]
            while len(set(tx_pos)) != len(tx_pos):
                tx_pos = np.random.permutation(np.arange(self.pos_max + 1))[:self.antenna_array.num_tx]
            self.positions[i, :self.antenna_array.num_tx] = np.sort(tx_pos)
        
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.full(self.num_particles, float('inf'))
    
    def fitness_function(self, particle_position):
        """
        Fitness function with sidelobe level optimization
        
        Returns:
        fitness: Fitness value (lower is better)
        """
        # Separate transmit and receive antenna positions
        tx_positions = particle_position[:self.antenna_array.num_tx]
        rx_positions = particle_position[self.antenna_array.num_rx:]
        
        # Check constraints
        violation_score = self.antenna_array.check_constraints(tx_positions, rx_positions)
        if violation_score > 0:
            return violation_score + 1000
        
        # Calculate virtual array
        virtual_positions = self.antenna_array.calculate_virtual_array(tx_positions, rx_positions)
        
        if len(virtual_positions) == 0:
            return float('inf')
        
        # Calculate HPBW
        hpbw, mainlobe_angle, theta, af = self.antenna_array.calculate_hpbw(virtual_positions)
        
        # Calculate SLL
        max_sll, _, _, _, _, _, _ = self.antenna_array.calculate_sll(virtual_positions)
        
        # Main fitness: difference between HPBW and target
        hpbw_error = abs(hpbw - self.antenna_array.target_hpbw)
        
        # Additional penalty: encourage main lobe at 0 degrees
        mainlobe_error = abs(mainlobe_angle)
        
        # SLL optimization: we want to minimize sidelobe level (make it more negative)
        sll_penalty = 0
        if max_sll > -np.inf:
            # We want SLL to be as low as possible (more negative)
            # So we penalize high SLL values
            sll_penalty = max(0, max_sll + 25)  # Target: SLL < -25dB
        
        # Total fitness with weights
        fitness = hpbw_error * 3 + mainlobe_error + sll_penalty * 5 + violation_score
        
        return fitness
    
    def optimize(self):
        """Execute PSO optimization"""
        self.initialize_particles()
        
        convergence_history = []
        best_history = []
        
        print("Starting PSO optimization with sidelobe minimization...")
        start_time = time.time()
        
        for iteration in range(self.max_iter):
            # Update inertia weight (linear decrease)
            self.w = 0.9 - 0.5 * (iteration / self.max_iter)
            
            for i in range(self.num_particles):
                # Calculate current fitness
                current_fitness = self.fitness_function(self.positions[i])
                
                # Update personal best
                if current_fitness < self.pbest_scores[i]:
                    self.pbest_scores[i] = current_fitness
                    self.pbest_positions[i] = self.positions[i].copy()
                
                # Update global best
                if current_fitness < self.gbest_score:
                    self.gbest_score = current_fitness
                    self.gbest_position = self.positions[i].copy()
            
            # Record history
            convergence_history.append(np.mean(self.pbest_scores))
            best_history.append(self.gbest_score)
            
            # Update velocities and positions
            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                # Velocity update
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.gbest_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                
                # Limit velocity
                self.velocities[i] = np.clip(self.velocities[i], -2, 2)
                
                # Position update
                self.positions[i] += self.velocities[i]
                
                # Ensure positions are integers
                self.positions[i] = np.round(self.positions[i])
                
                # Ensure positions are within range
                self.positions[i] = np.clip(self.positions[i], self.pos_min, self.pos_max)
                
                # Ensure transmit antennas don't overlap
                tx_pos = self.positions[i, :self.antenna_array.num_tx]
                if len(set(tx_pos)) != len(tx_pos):
                    # If overlapping, regenerate
                    tx_pos = np.random.permutation(np.arange(self.pos_max + 1))[:self.antenna_array.num_tx]
                    self.positions[i, :self.antenna_array.num_tx] = np.sort(tx_pos)
            
            # Print progress
            if (iteration + 1) % 20 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, Best Fitness: {self.gbest_score:.4f}")
        
        end_time = time.time()
        print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
        print(f"Best fitness: {self.gbest_score:.4f}")
        
        return convergence_history, best_history
    
    def get_best_solution(self):
        """Get the best solution"""
        if self.gbest_position is None:
            return None, None, None
        
        tx_positions = self.gbest_position[:self.antenna_array.num_tx]
        rx_positions = self.gbest_position[self.antenna_array.num_rx:]
        virtual_positions = self.antenna_array.calculate_virtual_array(tx_positions, rx_positions)
        
        return tx_positions, rx_positions, virtual_positions

def plot_and_save_results(tx_positions, rx_positions, virtual_positions, antenna_array, best_history, filename="antenna_optimization_results.png"):
    """Plot and save results as image"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Antenna layout
    ax1 = axes[0, 0]
    
    # Plot transmit antennas
    for i, pos in enumerate(tx_positions):
        ax1.plot([pos, pos], [0, 0.5], 'r-', linewidth=3)
        ax1.plot(pos, 0.5, 'ro', markersize=10, label='TX' if i == 0 else "")
    
    # Plot receive antennas
    for i, pos in enumerate(rx_positions):
        ax1.plot([pos, pos], [0, -0.5], 'b-', linewidth=3)
        ax1.plot(pos, -0.5, 'bs', markersize=10, label='RX' if i == 0 else "")
    
    # Plot virtual antennas
    for i, pos in enumerate(virtual_positions):
        ax1.plot(pos, 0, 'g^', markersize=6, alpha=0.5, label='Virtual' if i == 0 else "")
    
    ax1.set_xlabel('Position (half-wavelength units)')
    ax1.set_ylabel('Antenna Type')
    ax1.set_title('Antenna Array Layout')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-1, 1])
    
    # 2. Radiation pattern
    ax2 = axes[0, 1]
    theta = np.linspace(-90, 90, 1801)
    af = antenna_array.array_factor(virtual_positions, theta)
    af_normalized = af / np.max(af)
    af_db = 20 * np.log10(af_normalized + 1e-10)
    
    ax2.plot(theta, af_db, 'b-', linewidth=2)
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Normalized Pattern (dB)')
    ax2.set_title('MIMO Virtual Array Radiation Pattern')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-90, 90])
    ax2.set_ylim([-50, 5])
    
    # Calculate HPBW and SLL
    hpbw, mainlobe_angle, _, _ = antenna_array.calculate_hpbw(virtual_positions)
    max_sll, sll_angle, sll_value, _, _, left_min_angle, right_min_angle = antenna_array.calculate_sll(virtual_positions)
    
    # Mark HPBW
    ax2.axvline(x=mainlobe_angle, color='r', linestyle='--', alpha=0.5, label=f'Mainlobe: {mainlobe_angle:.1f}°')
    ax2.axhline(y=-3, color='g', linestyle='--', alpha=0.5, label='-3 dB')
    
    # Mark main lobe boundaries (first minima)
    ax2.axvline(x=left_min_angle, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Mainlobe left: {left_min_angle:.1f}°')
    ax2.axvline(x=right_min_angle, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Mainlobe right: {right_min_angle:.1f}°')
    
    # Shade the main lobe region
    ax2.axvspan(left_min_angle, right_min_angle, alpha=0.1, color='yellow', label='Mainlobe region')
    
    # Also get all sidelobe peaks for display
    sll_angles, sll_values = antenna_array.calculate_all_sidelobes(virtual_positions)
    
    if not np.isinf(max_sll):
        # Mark the maximum sidelobe
        ax2.plot(sll_angle, sll_value, 'r*', markersize=15, label=f'Max SLL: {max_sll:.1f} dB')
        ax2.text(sll_angle, sll_value + 2, f'{sll_value:.1f} dB', 
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Also mark other sidelobe peaks
        if len(sll_angles) > 0:
            ax2.plot(sll_angles, sll_values, 'r*', markersize=8, alpha=0.7)
            # Annotate the 3 highest sidelobe peaks
            if len(sll_values) > 0:
                sorted_indices = np.argsort(sll_values)[-3:][::-1]
                for i, idx in enumerate(sorted_indices):
                    if i < 3 and sll_angles[idx] != sll_angle:  # Don't annotate the same point twice
                        ax2.text(sll_angles[idx], sll_values[idx] + 1.5, f'{sll_values[idx]:.1f} dB', 
                                ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax2.legend(loc='upper right', fontsize=9)
    
    # Add HPBW and SLL annotation
    ax2.text(0.02, 0.98, f'HPBW: {hpbw:.2f}°\nMax SLL: {max_sll:.1f} dB\nMainlobe: {left_min_angle:.1f}° to {right_min_angle:.1f}°', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 3. Convergence curve
    ax3 = axes[1, 0]
    if best_history is not None and len(best_history) > 0:
        iterations = np.arange(1, len(best_history) + 1)
        ax3.semilogy(iterations, best_history, 'b-', linewidth=2)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Best Fitness (log scale)')
        ax3.set_title('PSO Convergence Curve')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No convergence data', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax3.transAxes)
        ax3.set_title('PSO Convergence Curve')
    
    # 4. Virtual array distribution
    ax4 = axes[1, 1]
    
    # Plot transmit and receive antenna positions
    for i, pos in enumerate(tx_positions):
        ax4.plot(pos, 1, 'ro', markersize=12, label='TX' if i == 0 else "")
    
    for i, pos in enumerate(rx_positions):
        ax4.plot(pos, 0.5, 'bs', markersize=10, label='RX' if i == 0 else "")
    
    # Plot virtual antenna positions
    for i, pos in enumerate(virtual_positions):
        ax4.plot(pos, 0, 'g^', markersize=8, label='Virtual' if i == 0 else "")
    
    # Mark aperture
    if len(virtual_positions) > 0:
        aperture = virtual_positions[-1] - virtual_positions[0]
        ax4.annotate(f'Aperture: {aperture:.1f} λ/2', 
                    xy=(virtual_positions[0], -0.2),
                    xytext=(virtual_positions[0] + aperture/2, -0.5),
                    arrowprops=dict(arrowstyle='->', lw=1.5),
                    fontsize=10)
    
    ax4.set_xlabel('Position (half-wavelength units)')
    ax4.set_ylabel('Antenna Type')
    ax4.set_title('Virtual Array Distribution')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([-0.6, 1.2])
    
    # Add information text
    info_text = f"TX positions: {tx_positions}\nRX positions: {rx_positions}\n"
    info_text += f"Virtual elements: {len(virtual_positions)}\n"
    info_text += f"Unique virtual elements: {len(set(virtual_positions))}\n"
    info_text += f"HPBW: {hpbw:.2f}°\n"
    info_text += f"Mainlobe region: {left_min_angle:.1f}° to {right_min_angle:.1f}°\n"
    info_text += f"Max SLL: {max_sll:.1f} dB at {sll_angle:.1f}°"
    
    plt.figtext(0.02, 0.02, info_text, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle('MIMO Antenna Array Optimization Results', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Results saved to {filename}")
    
    # Also save detailed pattern plot
    save_detailed_pattern(tx_positions, rx_positions, virtual_positions, antenna_array, 
                         filename.replace('.png', '_pattern.png'))
    
    plt.show()
    
    return fig

def save_detailed_pattern(tx_positions, rx_positions, virtual_positions, antenna_array, filename="detailed_pattern.png"):
    """Save detailed radiation pattern with focus on sidelobes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Full pattern
    theta = np.linspace(-90, 90, 1801)
    af = antenna_array.array_factor(virtual_positions, theta)
    af_normalized = af / np.max(af)
    af_db = 20 * np.log10(af_normalized + 1e-10)
    
    ax1.plot(theta, af_db, 'b-', linewidth=2)
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Normalized Pattern (dB)')
    ax1.set_title('Full Radiation Pattern')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-90, 90])
    ax1.set_ylim([-50, 5])
    
    # Calculate HPBW and SLL
    hpbw, mainlobe_angle, _, _ = antenna_array.calculate_hpbw(virtual_positions)
    max_sll, sll_angle, sll_value, _, _, left_min_angle, right_min_angle = antenna_array.calculate_sll(virtual_positions)
    sll_angles, sll_values = antenna_array.calculate_all_sidelobes(virtual_positions)
    
    ax1.axvline(x=mainlobe_angle, color='r', linestyle='--', alpha=0.5, label=f'Mainlobe: {mainlobe_angle:.1f}°')
    ax1.axhline(y=-3, color='g', linestyle='--', alpha=0.5, label='-3 dB')
    
    # Mark main lobe boundaries
    ax1.axvline(x=left_min_angle, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Mainlobe left: {left_min_angle:.1f}°')
    ax1.axvline(x=right_min_angle, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Mainlobe right: {right_min_angle:.1f}°')
    ax1.axvspan(left_min_angle, right_min_angle, alpha=0.1, color='yellow', label='Mainlobe region')
    
    if not np.isinf(max_sll):
        ax1.plot(sll_angle, sll_value, 'r*', markersize=15, label=f'Max SLL: {max_sll:.1f} dB')
        if len(sll_angles) > 0:
            ax1.plot(sll_angles, sll_values, 'r*', markersize=8, alpha=0.7)
    
    ax1.legend(loc='upper right', fontsize=9)
    
    # Zoomed pattern around main lobe
    zoom_range = 40
    ax2.plot(theta, af_db, 'b-', linewidth=2)
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Normalized Pattern (dB)')
    ax2.set_title(f'Zoomed Pattern (±{zoom_range}°)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-zoom_range, zoom_range])
    ax2.set_ylim([-50, 5])
    
    ax2.axvline(x=mainlobe_angle, color='r', linestyle='--', alpha=0.5, label=f'Mainlobe: {mainlobe_angle:.1f}°')
    ax2.axhline(y=-3, color='g', linestyle='--', alpha=0.5, label='-3 dB')
    
    # Mark main lobe boundaries if within zoom range
    if np.abs(left_min_angle) <= zoom_range:
        ax2.axvline(x=left_min_angle, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Left boundary: {left_min_angle:.1f}°')
    if np.abs(right_min_angle) <= zoom_range:
        ax2.axvline(x=right_min_angle, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Right boundary: {right_min_angle:.1f}°')
    
    if not np.isinf(max_sll) and np.abs(sll_angle) <= zoom_range:
        ax2.plot(sll_angle, sll_value, 'r*', markersize=15, label=f'Max SLL: {max_sll:.1f} dB')
    
    # Filter sidelobes within zoom range
    if len(sll_angles) > 0:
        within_range = np.abs(sll_angles) <= zoom_range
        if np.any(within_range):
            ax2.plot(sll_angles[within_range], sll_values[within_range], 'r*', markersize=8, alpha=0.7)
    
    ax2.legend(loc='upper right', fontsize=9)
    
    # Add statistics
    stats_text = f"HPBW: {hpbw:.2f}°\nMax SLL: {max_sll:.1f} dB at {sll_angle:.1f}°\n"
    stats_text += f"Mainlobe region: {left_min_angle:.1f}° to {right_min_angle:.1f}°\n"
    stats_text += f"TX: {tx_positions}\nRX: {rx_positions}"
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle('Detailed Radiation Pattern Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function"""
    # Initialize MIMO antenna array
    mimo_array = MIMOAntennaArray(
        num_tx=3,
        num_rx=4,
        virtual_aperture=20,
        target_hpbw=4.0
    )
    
    # Initialize PSO optimizer
    pso_optimizer = PSOOptimizer(
        antenna_array=mimo_array,
        num_particles=50,
        max_iter=200
    )
    
    # Execute optimization
    print("=" * 60)
    print("MIMO Antenna Array Optimization with Sidelobe Minimization")
    print("=" * 60)
    print(f"Number of TX antennas: {mimo_array.num_tx}")
    print(f"Number of RX antennas: {mimo_array.num_rx}")
    print(f"Virtual aperture: {mimo_array.virtual_aperture} half-wavelengths")
    print(f"Target HPBW: {mimo_array.target_hpbw} degrees")
    print("=" * 60)
    
    convergence_history, best_history = pso_optimizer.optimize()
    
    # Get optimal solution
    tx_positions, rx_positions, virtual_positions = pso_optimizer.get_best_solution()
    
    if tx_positions is not None:
        print("\n" + "=" * 60)
        print("OPTIMAL SOLUTION FOUND")
        print("=" * 60)
        print(f"TX positions (half-wavelengths): {tx_positions}")
        print(f"RX positions (half-wavelengths): {rx_positions}")
        print(f"Virtual array positions (half-wavelengths):")
        print(f"  {virtual_positions}")
        
        # Calculate performance metrics
        if len(virtual_positions) > 0:
            virtual_aperture = virtual_positions[-1] - virtual_positions[0]
        else:
            virtual_aperture = 0
            
        hpbw, mainlobe_angle, theta, af = mimo_array.calculate_hpbw(virtual_positions)
        max_sll, sll_angle, sll_value, _, _, left_min_angle, right_min_angle = mimo_array.calculate_sll(virtual_positions)
        
        print(f"\nPerformance Metrics:")
        print(f"  Virtual aperture: {virtual_aperture:.2f} half-wavelengths")
        print(f"  HPBW: {hpbw:.2f} degrees")
        print(f"  Mainlobe direction: {mainlobe_angle:.2f} degrees")
        print(f"  Mainlobe region: {left_min_angle:.1f}° to {right_min_angle:.1f}°")
        print(f"  Maximum sidelobe level: {max_sll:.1f} dB at {sll_angle:.1f}°")
        print(f"  Number of virtual elements: {len(virtual_positions)}")
        print(f"  Number of unique virtual elements: {len(set(virtual_positions))}")
        
        # Check constraints
        violation_score = mimo_array.check_constraints(tx_positions, rx_positions)
        if violation_score == 0:
            print("  All constraints satisfied!")
        else:
            print(f"  Constraints violated with score: {violation_score}")
        
        print("=" * 60)
        
        # Plot and save results
        print("\nGenerating and saving plots...")
        fig = plot_and_save_results(tx_positions, rx_positions, virtual_positions, 
                                   mimo_array, best_history, "antenna_optimization_results.png")
        
        # Save antenna positions to text file
        with open("antenna_positions.txt", "w") as f:
            f.write("MIMO Antenna Array Optimization Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"TX positions (half-wavelengths): {tx_positions}\n")
            f.write(f"RX positions (half-wavelengths): {rx_positions}\n")
            f.write(f"Virtual aperture: {virtual_aperture:.2f} half-wavelengths\n")
            f.write(f"HPBW: {hpbw:.2f} degrees\n")
            f.write(f"Mainlobe direction: {mainlobe_angle:.2f} degrees\n")
            f.write(f"Mainlobe region: {left_min_angle:.1f}° to {right_min_angle:.1f}°\n")
            f.write(f"Maximum sidelobe level: {max_sll:.1f} dB at {sll_angle:.1f}°\n")
            f.write(f"Number of virtual elements: {len(virtual_positions)}\n")
            f.write(f"Number of unique virtual elements: {len(set(virtual_positions))}\n")
        
        print("Antenna positions saved to 'antenna_positions.txt'")
    else:
        print("No feasible solution found. Try increasing the number of iterations or particles.")

if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    random.seed(42)
    
    # Run main program
    main()
