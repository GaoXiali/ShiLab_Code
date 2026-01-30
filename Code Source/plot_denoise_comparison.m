function plot_denoise_comparison(raw, clean, frame_idx, channel_idx, fs)
    % plot_denoise_comparison: 对比滤波前后的波形、频谱和图像
    % 输入: 
    %   raw: 原始数据 [1024*4096*n]
    %   clean: 去噪后数据 [1024*4096*n]
    %   frame_idx: 想要查看的帧索引
    %   channel_idx: 想要查看的换能器通道索引 (1-1024)
    %   fs: 采样频率 (40e6)

    % 提取特定通道数据
    sig_raw = double(raw(channel_idx, :, frame_idx));
    sig_clean = double(clean(channel_idx, :, frame_idx));
    t = (0:length(sig_raw)-1) / fs * 1e6; % 时间轴 (us)

    % 计算频谱 (FFT)
    L = length(sig_raw);
    f = fs * (0:(L/2)) / L / 1e6; % 频率轴 (MHz)
    fft_raw = abs(fft(sig_raw)/L);
    fft_clean = abs(fft(sig_clean)/L);

    figure('Color', 'w', 'Name', ['Denoise Analysis - Frame ', num2str(frame_idx)]);

    % --- 1. 时域波形对比 ---
    subplot(2, 2, 1);
    plot(t, sig_raw, 'Color', [0.7 0.7 0.7], 'DisplayName', 'Raw Signal'); hold on;
    plot(t, sig_clean, 'b', 'LineWidth', 1, 'DisplayName', 'Filtered');
    title(['Time Domain (Channel ', num2str(channel_idx), ')']);
    xlabel('Time (\mu s)'); ylabel('Amplitude');
    legend; grid on;

    % --- 2. 频域特性对比 ---
    subplot(2, 2, 2);
    semilogy(f, fft_raw(1:L/2+1), 'Color', [0.7 0.7 0.7]); hold on;
    semilogy(f, fft_clean(1:L/2+1), 'r', 'LineWidth', 1);
    title('Power Spectrum (FFT)');
    xlabel('Frequency (MHz)'); ylabel('Magnitude');
    xlim([0 20]); grid on;
    legend('Raw Spectrum', 'Filtered Spectrum');

    % --- 3. 原始 Sinogram 图像 ---
    subplot(2, 2, 3);
    imagesc(t, 1:size(raw,1), raw(:,:,frame_idx));
    colormap(gray);
    title('Raw Sinogram (B-scan)');
    xlabel('Time (\mu s)'); ylabel('Channel Index');
    caxis([mean(sig_raw)-3*std(sig_raw), mean(sig_raw)+3*std(sig_raw)]); % 增强对比度

    % --- 4. 去噪后 Sinogram 图像 ---
    subplot(2, 2, 4);
    imagesc(t, 1:size(clean,1), clean(:,:,frame_idx));
    title('Denoised Sinogram (B-scan)');
    xlabel('Time (\mu s)'); ylabel('Channel Index');
    caxis([mean(sig_clean)-3*std(sig_clean), mean(sig_clean)+3*std(sig_clean)]);
    
    linkaxes([subplot(2,2,3), subplot(2,2,4)], 'xy'); % 同步缩放
end