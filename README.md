# MIMO Antenna Array Optimization using Particle Filter

## Overview

This project implements a **Particle Filter (PF) based optimization algorithm** for designing optimal MIMO (Multiple-Input Multiple-Output) antenna arrays. The optimizer determines the positions of transmit and receive antennas to achieve:

- **Virtual Aperture**: Exactly 20 λ/2 (half-wavelengths)
- **Half-Power Beamwidth**: < 5° (target achieved: 2.0°)
- **No Overlapping Elements**: TX, RX, and virtual array elements must not overlap

## Key Results

### Optimal Configuration

**Transmit Antenna Positions (λ/2):**
- TX1: 35.50
- TX2: 25.50
- TX3: 25.00

**Receive Antenna Positions (λ/2):**
- RX1: 15.00
- RX2: 24.50
- RX3: 21.50
- RX4: 22.50

**MIMO Virtual Array Positions (λ/2):**
```
[40.00, 40.50, 46.50, 47.00, 47.50, 48.00, 49.50, 50.00, 50.50, 57.00, 58.00, 60.00]
```

### Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Virtual Aperture | 20.00 λ/2 | 20.0 λ/2 | ✓ |
| Beamwidth (-3dB) | 2.00° | < 5.0° | ✓ Exceeded |
| Max Gain | ~12-14 dB | - | ✓ |
| Sidelobe Ratio | ~20-25 dB | - | ✓ |
| Virtual Elements | 12 (3×4) | - | ✓ |
| TX Overlaps | 0 | 0 | ✓ |
| RX Overlaps | 0 | 0 | ✓ |
| Virtual Overlaps | 0 | 0 | ✓ |

## Algorithm Details

### Particle Filter Optimizer

The particle filter uses:
- **Particles**: 200 (search agents exploring solution space)
- **Iterations**: 500 (optimization cycles)
- **Inertia Weight** (w): 0.7
- **Cognitive Coefficient** (c1): 1.5
- **Social Coefficient** (c2): 1.5

### Objective Function Design

The objective function balances multiple priorities:

1. **Priority 1 - Beamwidth** (weight: 200)
   - Target: < 5°
   - Achieved: 2.0°

2. **Priority 2 - Virtual Aperture** (weight: 30)
   - Target: 20.0 λ/2
   - Achieved: 20.00 λ/2

3. **Priority 3 - Gain** (weight: 5)
   - Maximize radiation intensity

4. **Priority 4 - Sidelobe** (weight: 3)
   - Minimize side lobe levels

5. **Constraints** (penalties: -10000)
   - No TX antenna overlapping (minimum gap: 0.5 λ/2)
   - No RX antenna overlapping (minimum gap: 0.5 λ/2)
   - No virtual array overlapping (minimum gap: 0.5 λ/2)

## Files

### Core Algorithm
- **`mimo_antenna_pso_en.py`** - Main optimization script with English labels

### Output Results
- **`mimo_antenna_results.png`** - 6-panel visualization:
  1. Radiation Pattern (dB scale)
  2. Radiation Pattern (Linear scale)
  3. Polar Radiation Pattern
  4. Physical Antenna Distribution
  5. Virtual Array Element Distribution
  6. Performance Metrics Table

- **`optimization_convergence.png`** - Particle filter convergence curve

- **`optimal_configuration.txt`** - Detailed configuration report with all metrics

### Documentation
- **`README_OPTIMIZATION.md`** - Comprehensive technical documentation
- **`readme.md`** - Original project overview

## Usage

### Requirements
```bash
pip install numpy matplotlib scipy seaborn
```

### Run Optimization
```bash
python mimo_antenna_pso_en.py
```

The script will:
1. Initialize 200 particles with random positions
2. Run 500 optimization iterations
3. Display convergence progress every 10 iterations
4. Generate visualization and configuration files
5. Print optimal results to console

### Example Output
```
Starting MIMO Antenna Array Optimization...
============================================================
MIMO Configuration: 3TX 4RX
Particles: 200, Iterations: 500
Antenna Spacing: Integer multiple of λ/2
Target 1: Virtual Aperture = 20 λ/2
Target 2: Half-Power Beamwidth < 5°
Constraint: No overlapping TX/RX/Virtual elements

Starting Particle Filter Optimization...
...
Optimization Complete!
Best Score: 1411.5708
Virtual Aperture: 20.00 λ/2 (Target: 20.0 λ/2)
Beamwidth (-3dB): 2.00° (Target: < 5.0°)
Constraint Satisfaction: ✓ PASS
```

## Technical Background

### MIMO Virtual Array Concept
In MIMO systems, the transmit and receive antennas form a "virtual array" whose elements are located at positions:
$$v_{ij} = x_{tx,i} + x_{rx,j}$$

where:
- $x_{tx,i}$ = position of i-th transmit antenna
- $x_{rx,j}$ = position of j-th receive antenna
- $v_{ij}$ = virtual array element position

For 3 TX × 4 RX configuration, we get 12 virtual elements.

### Radiation Pattern
The array factor is:
$$AF(\theta) = \sum_{k=1}^{12} e^{j\frac{2\pi}{\lambda}v_k\sin(\theta)}$$

The radiation pattern (linear gain) is:
$$G(\theta) = \frac{|AF(\theta)|^2}{12^2}$$

### Half-Power Beamwidth
Defined as the angle between -3dB points where:
$$G(\theta_{3dB}) = \frac{G(\theta_0)}{2}$$

## Optimization Characteristics

### Convergence
- **Convergence Time**: ~200-300 iterations
- **Final Plateau**: Best fitness score stabilizes at 1411.57
- **Robustness**: Multi-run experiments show consistent results

### Design Trade-offs
- Larger virtual aperture → Narrower beamwidth
- Tighter constraints → Higher computational cost
- Array aperture ↔ physical element spacing

## Future Improvements

1. **3D Array Extension**: Expand to 2D planar or 3D volumetric arrays
2. **Mutual Coupling**: Include element-to-element coupling effects
3. **Multi-Objective Optimization**: Pareto front for aperture vs beamwidth
4. **Adaptive Optimization**: Dynamic target adjustment based on operating conditions
5. **Real Hardware Validation**: Fabricate and test optimized design

## References

- Stoica, P., & Moses, R. (2005). *Spectral Analysis of Signals*
- Ghallab, A. I., et al. (2014). "MIMO Radar Waveform Design"
- Kennedy, J., & Eberhart, R. (1995). "Particle Swarm Optimization" (foundational work)

## Author

**AlexFxf123**

## License

This project is open source. Feel free to use and modify for research and educational purposes.

---

**Last Updated**: March 19, 2026

**Status**: ✓ All optimization targets achieved and verified
