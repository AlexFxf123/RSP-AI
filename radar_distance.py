import numpy as np
import matplotlib.pyplot as plt
import math
import os
import time

class FMCWRadar:
    """FMCW雷达探测距离计算类（使用最新公式）"""
    
    def __init__(self):
        """初始化雷达参数"""
        # 雷达系统参数
        self.pt_dbm = 12.0  # 发射功率，dBm
        self.gain_tx_db = 20.0  # 发射天线增益，dB
        self.gain_rx_db = 20.0  # 接收天线增益，dB
        self.freq_hz = 77.0e9  # 频率，Hz
        self.t_mod_s = 20.0e-6  # 调频时间，s
        self.gs_db = 30.0  # 信号处理增益，dB（从32修改为30）
        
        # 环境参数
        self.temp_k = 300.0  # 工作温度，K
        self.noise_figure_db = 12.0  # 接收机噪声系数，dB
        self.snr_min_db = 12.0  # 最小检测信噪比，dB
        
        # 目标参数
        self.rcs_m2 = 0.01  # 目标RCS，m²
        
        # 物理常数
        self.c = 3.0e8  # 光速，m/s
        self.k = 1.380649e-23  # 玻尔兹曼常数，J/K
        self.pi = math.pi
    
    def dbm_to_watt(self, dbm):
        """dBm转换为瓦特"""
        return 1e-3 * 10**(dbm/10.0)
    
    def db_to_linear(self, db):
        """dB转换为线性值"""
        return 10**(db/10.0)
    
    def calculate_wavelength(self):
        """计算波长"""
        return self.c / self.freq_hz
    
    def calculate_range(self, rcs=None):
        """
        基于最新雷达方程计算探测距离
        
        最新公式：R_max^4 = (P_t * G_t * G_r * λ^2 * σ * G_s * t_mod) / ((4π)^3 * k * T * F * (S/N)_{min})
        
        参数:
        rcs: 目标RCS，m²。如果为None，使用实例的rcs_m2
        
        返回:
        探测距离，米
        """
        if rcs is None:
            rcs = self.rcs_m2
        
        # 参数转换
        pt_w = self.dbm_to_watt(self.pt_dbm)  # 发射功率，W
        g_t = self.db_to_linear(self.gain_tx_db)  # 发射天线增益（线性）
        g_r = self.db_to_linear(self.gain_rx_db)  # 接收天线增益（线性）
        wavelength = self.calculate_wavelength()  # 波长，m
        t = self.temp_k  # 温度，K
        f_linear = self.db_to_linear(self.noise_figure_db)  # 噪声系数（线性）
        snr_min_linear = self.db_to_linear(self.snr_min_db)  # 最小信噪比（线性）
        gs_linear = self.db_to_linear(self.gs_db)  # 信号处理增益（线性）
        t_mod = self.t_mod_s  # 调频时间，s
        
        # 最新雷达方程：R^4 = (Pt * G_t * G_r * λ^2 * σ * G_s * t_mod) / ((4π)^3 * k * T * F * SNR_min)
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
        """计算最大不模糊距离"""
        return self.c * self.t_mod_s / 2
    
    def print_parameters(self):
        """打印雷达参数"""
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
        print("FMCW雷达参数配置表（最新公式）")
        print("=" * 90)
        print(f"{'参数':<25} {'值':<20} {'单位':<15} {'说明'}")
        print("-" * 90)
        print(f"{'目标RCS (σ)':<25} {self.rcs_m2:<20.4f} {'m²':<15} 雷达散射截面积")
        print(f"{'发射功率 (P_t)':<25} {self.pt_dbm:<20.1f} {'dBm':<15} 线性值: {pt_w:.3e} W")
        print(f"{'发射天线增益 (G_t)':<25} {self.gain_tx_db:<20.1f} {'dB':<15} 线性值: {g_t_linear:.3f}")
        print(f"{'接收天线增益 (G_r)':<25} {self.gain_rx_db:<20.1f} {'dB':<15} 线性值: {g_r_linear:.3f}")
        print(f"{'信号处理增益 (G_s)':<25} {self.gs_db:<20.1f} {'dB':<15} 线性值: {gs_linear:.3f}")
        print(f"{'频率 (f)':<25} {self.freq_hz/1e9:<20.1f} {'GHz':<15} 波长: {wavelength*100:.3f} cm")
        print(f"{'调频时间 (T_mod)':<25} {self.t_mod_s*1e6:<20.1f} {'μs':<15} 20 μs")
        print(f"{'温度 (T)':<25} {self.temp_k:<20.0f} {'K':<15} 工作温度")
        print(f"{'噪声系数 (F)':<25} {self.noise_figure_db:<20.1f} {'dB':<15} 线性值: {nf_linear:.3f}")
        print(f"{'最小信噪比 (S/N_min)':<25} {self.snr_min_db:<20.1f} {'dB':<15} 线性值: {snr_min_linear:.3f}")
        print(f"{'最大不模糊距离':<25} {range_unamb:<20.1f} {'m':<15} R_unamb = c×T_mod/2")
        print("-" * 90)
        print(f"{'估计探测距离':<25} {range_m:<20.2f} {'m':<15} 约 {range_m/1000:.2f} km")
        print("=" * 90)
        print()


def plot_range_vs_rcs(radar, rcs_min=0.001, rcs_max=0.1, num_points=100, save_fig=True):
    """
    绘制探测距离随RCS变化图
    """
    # 生成RCS范围
    rcs_values = np.linspace(rcs_min, rcs_max, num_points)
    range_values = np.zeros_like(rcs_values)
    
    # 计算每个RCS对应的探测距离
    for i, rcs in enumerate(rcs_values):
        range_values[i] = radar.calculate_range(rcs)
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 探测距离随RCS变化
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(rcs_values, range_values, 'b-', linewidth=2)
    ax1.set_xlabel('Target RCS (m²)', fontsize=12)
    ax1.set_ylabel('Detection Range (m)', fontsize=12)
    ax1.set_title('Detection Range vs Target RCS', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # 标记当前配置对应的点
    current_range = radar.calculate_range(radar.rcs_m2)
    ax1.plot(radar.rcs_m2, current_range, 'ro', markersize=10, 
             label=f'RCS={radar.rcs_m2} m²\nR={current_range:.1f} m')
    ax1.legend(fontsize=10)
    
    # 对数坐标下的探测距离
    ax2 = plt.subplot(2, 2, 2)
    ax2.loglog(rcs_values, range_values, 'r-', linewidth=2)
    ax2.set_xlabel('Target RCS (m²)', fontsize=12)
    ax2.set_ylabel('Detection Range (m)', fontsize=12)
    ax2.set_title('Detection Range vs Target RCS (Log-Log Scale)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    # 距离随调频时间变化
    ax3 = plt.subplot(2, 2, 3)
    t_mod_values = np.linspace(1, 100, 50)  # 1 到 100 μs
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
    
    # 距离随天线增益变化（假设G_t = G_r）
    ax4 = plt.subplot(2, 2, 4)
    gain_values = np.linspace(0, 30, 50)  # 0 到 30 dB
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
    
    # 保存图片
    if save_fig:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"detection_range_vs_rcs_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图片已保存为: {filename}")
    
    plt.show()
    plt.close()
    
    return rcs_values, range_values


def plot_range_vs_processing_gain_and_power(radar, save_fig=True):
    """绘制探测距离随信号处理增益和发射功率变化图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 距离随信号处理增益变化
    gs_values = np.linspace(0, 50, 51)  # 0 到 50 dB
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
    
    # 距离随发射功率变化
    power_values = np.linspace(-20, 20, 41)  # -20 到 20 dBm
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
    
    # 保存图片
    if save_fig:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"detection_range_vs_gain_power_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图片已保存为: {filename}")
    
    plt.show()
    plt.close()


def plot_range_vs_frequency_and_snr(radar, save_fig=True):
    """绘制探测距离随频率和信噪比变化图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 距离随频率变化
    freq_values = np.linspace(24, 120, 50)  # 24 到 120 GHz
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
    
    # 距离随最小信噪比变化
    snr_values = np.linspace(0, 20, 41)  # 0 到 20 dB
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
    
    # 保存图片
    if save_fig:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"detection_range_vs_freq_snr_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图片已保存为: {filename}")
    
    plt.show()
    plt.close()


def main():
    """主函数"""
    print("FMCW雷达探测距离计算程序（最新公式）")
    print("=" * 60)
    print("最新公式：R_max^4 = (P_t * G_t * G_r * λ^2 * σ * G_s * t_mod) / ((4π)^3 * k * T * F * (S/N)_{min})")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = "radar_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.chdir(output_dir)
    print(f"图片将保存到目录: {output_dir}")
    
    # 创建雷达实例
    radar = FMCWRadar()
    
    # 打印参数
    radar.print_parameters()
    
    # 绘制探测距离随RCS变化图
    print("生成探测距离随RCS变化图...")
    plot_range_vs_rcs(radar, save_fig=True)
    
    # 绘制探测距离随信号处理增益和发射功率变化图
    print("生成探测距离随信号处理增益和发射功率变化图...")
    plot_range_vs_processing_gain_and_power(radar, save_fig=True)
    
    # 绘制探测距离随频率和信噪比变化图
    print("生成探测距离随频率和信噪比变化图...")
    plot_range_vs_frequency_and_snr(radar, save_fig=True)
    
    # 计算不同RCS下的探测距离示例
    print("\n不同RCS目标探测距离示例:")
    print("-" * 60)
    rcs_examples = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]  # m²
    
    for rcs in rcs_examples:
        distance = radar.calculate_range(rcs)
        print(f"RCS = {rcs:6.3f} m²: 探测距离 = {distance:8.2f} m ({distance/1000:6.2f} km)")
    
    # 参数敏感性分析
    print("\n参数敏感性分析:")
    print("-" * 60)
    
    # 测试不同信号处理增益
    test_gs = [0, 10, 20, 30, 40, 50]  # dB
    print("\n不同信号处理增益下的探测距离:")
    for gs in test_gs:
        radar_temp = FMCWRadar()
        radar_temp.gs_db = gs
        distance = radar_temp.calculate_range()
        print(f"G_s = {gs:2d} dB: 探测距离 = {distance:7.2f} m")
    
    # 测试不同天线增益
    test_gain = [0, 10, 20, 30]  # dB
    print("\n不同天线增益下的探测距离 (G_t = G_r):")
    for gain in test_gain:
        radar_temp = FMCWRadar()
        radar_temp.gain_tx_db = gain
        radar_temp.gain_rx_db = gain
        distance = radar_temp.calculate_range()
        print(f"G_t = G_r = {gain:2d} dB: 探测距离 = {distance:7.2f} m")
    
    # 测试不同调频时间
    test_t_mod = [10, 20, 50, 100]  # μs
    print("\n不同调频时间下的探测距离:")
    for t_mod in test_t_mod:
        radar_temp = FMCWRadar()
        radar_temp.t_mod_s = t_mod * 1e-6
        distance = radar_temp.calculate_range()
        print(f"T_mod = {t_mod:3d} μs: 探测距离 = {distance:7.2f} m")
    
    print(f"\n所有图片已保存到目录: {output_dir}")


if __name__ == "__main__":
    main()
