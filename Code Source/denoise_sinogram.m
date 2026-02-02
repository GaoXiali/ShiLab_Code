function clean_sinogram = denoise_sinogram(raw_sinogram)
    % denoise_sinogram: 针对光声sinogram数据的频域带通滤波
    % 输入: raw_sinogram [Channel*Sampling*Frame] [1024 * 4096 * n]
    % 输出: clean_sinogram [Channel*Sampling*Frame] [1024 * 4096 * n]

    %% 1. 参数设置
    fs = 40e6;              % 采样频率 40MHz
    fc = 3.6e6;             % 中心频率 3.6MHz
    fwhm_pct = 1.2;         % FWHM 带宽 120%
    
    % 计算截止频率 (Hz)
    % 有效带宽 = fc * fwhm_pct = 4.32 MHz
    % 频带范围 = [fc - BW/2, fc + BW/2]
    f_low = fc * (1 - fwhm_pct/2);
    f_high = fc * (1 + fwhm_pct/2);
    
    % 边界检查：确保低频不小于0，高频不超过奈奎斯特频率
    f_low = max(f_low, 100e3); % 至少预留100kHz滤除基线漂移
    f_nyquist = fs / 2;
    f_high = min(f_high, f_nyquist - 1e5);
    
    %% 2. 设计 Butterworth 滤波器
    % 归一化截止频率 (Wn = f / (fs/2))
    Wn = [f_low, f_high] / (fs / 2);
    order = 4; % 4阶兼顾过渡带陡峭度和计算性能
    [b, a] = butter(order, Wn, 'bandpass');
    
    %% 3. 执行滤波
    % 数据结构为 [阵元, 时间点, 帧数]
    % MATLAB 的 filtfilt 默认沿第一维度处理，我们的时间点在第二维度
    % 因此需要对第二维度（4096点）进行操作
    
    [num_elements, num_samples, num_frames] = size(raw_sinogram);
    clean_sinogram = zeros(size(raw_sinogram), 'like', raw_sinogram);
    
    fprintf('正在进行带通滤波 (%0.2f - %0.2f MHz)...\n', f_low/1e6, f_high/1e6);
    
    for k = 1:num_frames
        % 提取当前帧 [1024 * 4096]
        current_frame = raw_sinogram(:, :, k);
        
        % 转置以使时间轴变为第一维，提速处理
        % 处理后 [4096 * 1024]
        temp = current_frame';
        
        % 执行零相位滤波
        temp_filtered = filtfilt(b, a, double(temp));
        
        % 转置回原始维度并存入结果
        clean_sinogram(:, :, k) = temp_filtered';
    end
    
    fprintf('去噪完成。\n');

    plot_denoise_comparison(raw_sinogram, clean_sinogram, 1, 512, fs); % 查看去噪前后效果对比
end