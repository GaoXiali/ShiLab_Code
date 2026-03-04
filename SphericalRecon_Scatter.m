% ************************* 3D reconstruction code ************************
%
% 本程序用于球阵光声/超声双模态探测器阵列数据3D重建
%
% Author：Xiali Gao; Hao Huang  Version：4.0  2026.1.30
%
% *************************************************************************

%% 初始化及加载文件
clc; 
% clearvars -except data; 
% clear;
% close all; 
gpuDevice(1).reset() % 重置GPU，释放所有显存

addpath 'D:\code\code_from_XiaLi';
addpath 'D:\code\code_from_XiaoYang\final_version'

folder_path = 'G:\26.02.02\z=5\';
fixMatlabFilenames(folder_path);%自动校正错误文件名
str_name = dir(fullfile(folder_path, '*_0.mat'));
[datax,DAQ_time_point] = func_3D_PACT_Data_Time_Read(folder_path,str_name(577).name);

% 根据表面信号判断区分超声帧和光声帧
frame1_val = max(sum(datax(:, 1:100, 1)));
frame2_val = max(sum(datax(:, 1:100, 2)));
offset = (frame1_val < frame2_val); 
pa_idx = (1 + offset) : 2 : size(datax, 3); % 光声帧索引
us_idx = (2 - offset) : 2 : size(datax, 3); % 超声帧索引

data = datax(:,:,pa_idx);%选择光声帧数
dataUS = permute(datax(:, :, us_idx), [2, 1, 3]);%选择超声帧数

Aline = mean(data,3);
Aline = squeeze(mean(Aline,1));
[~,DL1] = max(Aline(1:100));

detector=load('coordinate.txt'); % 加载探测器坐标文件
detector(:,1) = detector(:,1)+0.555;%坐标校正
detector(:,2) = detector(:,2)+0.39;

[Nelemt, Nsample, Nframe] = size(data); % 获取样本数和元素数

% figure(1);imagesc(-data(:,:,1));colormap gray;

folders = {'USresult', 'PAresult'}; % 将所有需要创建的文件夹放在这里

for i = 1:length(folders)
    % 修正：第二个参数必须是 'dir'
    if ~exist(folders{i}, 'dir') 
        mkdir(folders{i});
        fprintf('已创建文件夹: %s\n', folders{i});
    else
        fprintf('文件夹已存在: %s\n', folders{i});
    end
end

%% 系统参数设置
reconstruct_mode = 4; % 1: 单声速CUDA重建; 2: 单声速扫描遍历; 3：PA粒子遍历成像并统计FWHM 
                      % 4：外声速迭代; 5:单声速迭代 6:单声速旋转复合 7:双声速旋转复合 
                      % 8:单声速相干因子旋转复合 9:单/双声速旋转平移复合 
                      % 10:超声单声速重建  11: 超声单声速遍历 12:US粒子遍历成像并统计FWHM
                      % 13:光声固定视场物体平移复合

% 声速设置  
T = 22.8; % 水温
V_M = 1490;
V_M_Range = 1491:1:1510; % 单声速迭代范围
PA_FRAME_COMPOUND = 15;

% 超声参数设置
V_US = 1498.5;
US_FRAME_COMPOUND = 1;
Dynamic_Range = 30; % dB

Is_Gating = 0; % 1表示需要门控，0表示不需要门控,直接对前US_FRAME_COMPOUND帧进行复合
Is_Denoising = 0; % 1表示去噪，0表示不去噪，去除换能器接收带宽外的噪声，会一定程度增加计算量(目前仅修改了光声相关代码）

% VM_out = waterSoundSpeed(T); % 外声速（水），单位 m/s 
VM_out = 1482; % 外声速（水），单位 m/s  
VM_out_Range = 1475:0.5:1499; % 外声速迭代范围
VM_in = 1482; % 内声速，单位 m/s  
VM_in_Range = 1555:5:1699; % 内声速迭代范围

% 位移扫描参数
step_x = 1; %mm 平移复合位移数（文件数量方向）
step_y = 1; %mm 平移复合位移数(帧/文件数量方向）
step_length_x = 6; %mm 水平相邻两帧位移距离
step_length_y = 8; %mm 垂直相邻两帧位移距离
Nframex_scan = 7; %x方向扫描次数
Ndata = size(str_name,1)/4;%采集数据组数
Nframey_scan = floor(Ndata/Nframex_scan); %y方向扫描次数
GaussianMask_FWHM = 40; % 子图中的高斯掩膜尺寸，避免拼接出缝，小散射片用20，大散射片用40
 
% % % % % 图像重建的尺寸设置
% x_size = 50;
% y_size = 50;
% z_size = 20;
% resolution_factor = 10; % 分辨率因子

x_size = 2;
y_size = 2;
z_size = 2;
resolution_factor = 100; % 分辨率因子

% % US SCATTER
% center_x = 1.2;  % 中心坐标X
% center_y = 2.5; % 中心坐标Y
% center_z = -5;  % 中心坐标Z

% % US SCATTER
% center_x = 0.3;  % 中心坐标X
% center_y = -5.6; % 中心坐标Y
% center_z = -5;  % 中心坐标Z

% PA SCATTER
center_x = 0.5;  % 中心坐标X
center_y = 0.3; % 中心坐标Y
center_z = -6.2;  % 中心坐标Z

%椭球面参数设置
Ellipse.a = 15.0; % 椭球面x轴半径
Ellipse.b = 28.0;  % 椭球面y轴半径
Ellipse.c = 13.5; % 椭球面z轴半径
Ellipse.centerx = -2.7; % 椭球面中心坐标
Ellipse.centery = -1.1;
Ellipse.centerz = 7.2;

%系统参数
if Is_Denoising == 1 
    data = denoise_sinogram(data);%滤除换能器带宽外的噪声
    dataUS = denoise_sinogram(dataUS);
end

predelay = -DL1; % 预延迟设置
pa_data = -data;% 取负是为了让重建背景反色
fs = 40; % 声音采样频率 单位为MHz
R = 100; % 球形探测器阵列半径

% %对球阵传感器阵列插值p=] 
% [pa_data, detector] = interpolation(pa_data(:,:,1),detector);

% 计算像素点的具体位置
Npixel_x = x_size * resolution_factor+1;
Npixel_y = y_size * resolution_factor+1;
Npixel_z = z_size * resolution_factor+1;
x_range = ((1:Npixel_x)-(Npixel_x+1)/2)*x_size/(Npixel_x-1) + center_x;
y_range = ((1:Npixel_y)-(Npixel_y+1)/2)*y_size/(Npixel_y-1) + center_y;
z_range = ((1:Npixel_z)-(Npixel_z+1)/2)*z_size/(Npixel_z-1) + center_z;

% 创建三维网格坐标
[X_img, Y_img, Z_img] = meshgrid(x_range, y_range, z_range);

% % 坐标旋转(角度制,顺时针为负，逆时针为正）
theta_x = 0; % 以x轴为中心旋转 
theta_y = 0; % 以y轴为中心旋转 
theta_z = 48.38; % 以z轴为中心旋转 位移台与坐标轴夹角48.38

trans_x = 0;
trans_y = 0;
trans_z = 0;

rotate_x_mat = [1 0 0 0;0 cosd(theta_x) -sind(theta_x),0;0 sind(theta_x) cosd(theta_x) 0;0 0 0 1];
rotate_y_mat = [cosd(theta_y) 0 -sind(theta_y) 0;0 1 0,0;sind(theta_y) 0 cosd(theta_y) 0;0 0 0 1];
rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0;sind(theta_z) cosd(theta_z) 0,0;0 0 1 0;0 0 0 1];
trans_mat = [1 0 0 trans_x;0 1 0 trans_y;0 0 1 trans_z;0 0 0 1];
afine_mat = trans_mat*rotate_x_mat*rotate_y_mat*rotate_z_mat; %以原点为中心，先转z轴，再转y轴，再转x轴，最后平移

detector_new=[detector,detector(:,1)*0+1]*afine_mat'; 

% 获取并调整传感器位置，Z 坐标取负以调整坐标系统
x_sensor=detector_new(:,1);
y_sensor=detector_new(:,2);
z_sensor=-detector_new(:,3); % 注意此处z轴坐标反转

% 将传感器坐标移至GPU以加速计算
x_sensor = gpuArray(single(x_sensor));
y_sensor = gpuArray(single(y_sensor));
z_sensor = gpuArray(single(z_sensor));

% 将图像坐标移至GPU
X_img = gpuArray(single(X_img));
Y_img = gpuArray(single(Y_img));
Z_img = gpuArray(single(Z_img));
Points_img = cat(4,X_img,Y_img,Z_img);

% 回波信号数据处理的主循环
tic % 开始计时
switch reconstruct_mode

    case 1 %单声速重建
        for frame = 1%:Nframe %设置双声速重建区域时，最好使用双声速重建复合的第一帧确定参数
            tic
            pa_data_frame = gpuArray(single(pa_data(:,:,frame))); % [Nelemt x Nsample]
            Points_sensor_all = gpuArray(single([x_sensor,y_sensor,z_sensor]));% [Nelemt x 3]
             
            [pa_img, total_angle_weight] = SingleSpeedReconstraction_mex(Points_sensor_all, Points_img, pa_data_frame, single(fs), single(predelay), single(V_M), single(R));            % % 初始化图像缓冲区和角度权重
            disp(['frame : ',num2str(frame)]);
            toc
            pa_img1 = gather(pa_img);
            total_angle_weight = gather(total_angle_weight);
            pa_img2 = subplus(pa_img1)./total_angle_weight;
            imin=min(pa_img2,[],"all");
            imax=max(pa_img2,[],"all");
            
            figure(); set (gca,'position',[0.1,0.1,0.8,0.8]);
            subplot(131); imagesc(z_range, x_range, squeeze(max(pa_img2(:,:,:),[],1)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'tickdir', 'out'); 
            ylabel('X'); xlabel('Z'); title('XZ proj'); 
            subplot(133); imagesc(z_range, y_range, squeeze(max(pa_img2(:,:,:),[], 2)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'tickdir', 'out'); 
            ylabel('Y'); xlabel('Z'); title('YZ proj'); 
            subplot(132); imagesc(x_range, y_range, squeeze(max(pa_img2(:,:,:),[], 3)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; axis equal;
            ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out'); 
            % drawEllipsoidOverlay(Ellipse);%显示双声速范围

            filenameZX = sprintf('PAresult/Single Frame zx frame=%d,V_M=%.1f.png',frame, V_M);
            filenameZY = sprintf('PAresult/Single Frame zy frame=%d,V_M=%.1f.png',frame, V_M);
            filenameXY = sprintf('PAresult/Single Frame xy frame=%d,V_M=%.1f.png',frame, V_M);
            imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],1))), filenameZX);
            imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],2))), filenameZY);
            imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],3))), filenameXY);
        end

    case 2 %双声速cuda重建
        for frame = 145:145
            [datax,DAQ_time_point] = func_3D_PACT_Data_Time_Read(folder_path,str_name(1 + (frame-1)*4).name);

            % 根据表面信号判断区分超声帧和光声帧
            frame1_val = max(sum(datax(:, 1:100, 1)));
            frame2_val = max(sum(datax(:, 1:100, 2)));
            offset = (frame1_val < frame2_val); 
            pa_idx = (1 + offset) : 2 : size(datax, 3); % 光声帧索引
            us_idx = (2 - offset) : 2 : size(datax, 3); % 超声帧索引
            
            data = datax(:,:,pa_idx);%选择光声帧数
            
            Aline = mean(data,3);
            Aline = squeeze(mean(Aline,1));
            [~,DL1] = max(Aline(1:100));
     
            pa_data = -data;% 取负是为了让重建背景反色
            
            pa_data_frame = gpuArray(single(data(:,:,1))); % [Nelemt x Nsample]
            Points_sensor_all = gpuArray(single([x_sensor,y_sensor,z_sensor]));% [Nelemt x 3]
             
            % 调用 CUDA MEX 函数
            [pa_img, total_angle_weight] = SingleSpeedReconstraction_mex(Points_sensor_all, Points_img, pa_data_frame, single(fs), single(predelay), single(V_M), single(R));            % % 初始化图像缓冲区和角度权重

            pa_img1 = gather(pa_img);
            total_angle_weight = gather(total_angle_weight);
            
            % 归一化并应用非线性增强
            pa_img2 = subplus(pa_img1)./total_angle_weight;
            
            % 显示不同视角的图像
            imin=min(pa_img2,[],"all");
            imax=max(pa_img2,[],"all");
            
            figure(2); set (gca,'position',[0.1,0.1,0.8,0.8]);
            subplot(131); imagesc(z_range, x_range, squeeze(max(pa_img2(:,:,:),[],1))); 
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'tickdir', 'out'); 
            ylabel('X'); xlabel('Z'); title('XZ proj'); 
            subplot(133); imagesc(z_range, y_range, squeeze(max(pa_img2(:,:,:),[], 2))); 
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'tickdir', 'out'); 
            ylabel('Y'); xlabel('Z'); title('YZ proj'); 
            subplot(132); imagesc(x_range, y_range, squeeze(max(pa_img2(:,:,:),[], 3))); 
            axis equal tight; colormap gray; colorbar; axis equal;
            ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out'); 

            filenameZX = sprintf('PAresult/Scatter zx ScanNum=%d,V_M=%.1f.png',frame, V_M);
            filenameZY = sprintf('PAresult/Scatter zy ScanNum=%d,V_M=%.1f.png',frame, V_M);
            filenameXY = sprintf('PAresult/Scatter xy ScanNum=%d,V_M=%.1f.png',frame, V_M);
            imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],1))), filenameZX);
            imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],2))), filenameZY);
            imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],3))), filenameXY);

            disp(frame)
        end

    case 3  %数据遍历，批量计算FWHM
        fwhm_result = zeros(289,3);
        for frame = 145:145
            [datax,DAQ_time_point] = func_3D_PACT_Data_Time_Read(folder_path,str_name(1 + (frame-1)*4).name);
            
            % 根据表面信号判断区分超声帧和光声帧
            frame1_val = max(sum(datax(:, 1:100, 1)));
            frame2_val = max(sum(datax(:, 1:100, 2)));
            offset = (frame1_val < frame2_val); 
            pa_idx = (1 + offset) : 2 : size(datax, 3); % 光声帧索引
            us_idx = (2 - offset) : 2 : size(datax, 3); % 超声帧索引
            
            data = datax(:,:,pa_idx);%选择光声帧数
            dataUS = permute(datax(:, :, us_idx), [2, 1, 3]);%选择超声帧数
            
            Aline = mean(data,3);
            Aline = squeeze(mean(Aline,1));
            [~,DL1] = max(Aline(1:100));
          
            
            % % % % 图像重建的尺寸设置
            % x_size = 50;
            % y_size = 50;
            % z_size = 20;
            % resolution_factor = 10; % 分辨率因子
            
            x_size = 2.5;
            y_size = 2.5;
            z_size = 2;
            resolution_factor = 100; % 分辨率因子
            
            center_x = floor(frame/17) * (-5) + 0.5 + 40;  % 中心坐标X
            center_y = mod(frame-1,17) * 5 + 0.3 - 40; % 中心坐标Y
            center_z = -6.2;  % 中心坐标Z
            
            %系统参数
            if Is_Denoising == 1 
                data = denoise_sinogram(data);%滤除换能器带宽外的噪声
                dataUS = denoise_sinogram(dataUS);
            end
            
            predelay = -DL1; % 预延迟设置
            pa_data = -data;% 取负是为了让重建背景反色

            % 计算像素点的具体位置
            Npixel_x = x_size * resolution_factor+1;
            Npixel_y = y_size * resolution_factor+1;
            Npixel_z = z_size * resolution_factor+1;
            x_range = ((1:Npixel_x)-(Npixel_x+1)/2)*x_size/(Npixel_x-1) + center_x;
            y_range = ((1:Npixel_y)-(Npixel_y+1)/2)*y_size/(Npixel_y-1) + center_y;
            z_range = ((1:Npixel_z)-(Npixel_z+1)/2)*z_size/(Npixel_z-1) + center_z;
            
            % 创建三维网格坐标
            [X_img, Y_img, Z_img] = meshgrid(x_range, y_range, z_range);
            
            % 将图像坐标移至GPU
            X_img = gpuArray(single(X_img));
            Y_img = gpuArray(single(Y_img));
            Z_img = gpuArray(single(Z_img));
            Points_img = cat(4,X_img,Y_img,Z_img);
    
            tic
            pa_data_frame = gpuArray(single(pa_data(:,:,1))); % [Nelemt x Nsample]
            Points_sensor_all = gpuArray(single([x_sensor,y_sensor,z_sensor]));% [Nelemt x 3]
             
            [pa_img, total_angle_weight] = SingleSpeedReconstraction_mex(Points_sensor_all, Points_img, pa_data_frame, single(fs), single(predelay), single(V_M), single(R));            % % 初始化图像缓冲区和角度权重
            toc
            pa_img1 = gather(pa_img);
            total_angle_weight = gather(total_angle_weight);
            pa_img2 = subplus(pa_img1)./total_angle_weight;
            imin=min(pa_img2,[],"all");
            imax=max(pa_img2,[],"all");
            
            figure(7); set (gca,'position',[0.1,0.1,0.8,0.8]);
            subplot(131); imagesc(z_range, x_range, squeeze(max(pa_img2(:,:,:),[],1)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'tickdir', 'out'); 
            ylabel('X'); xlabel('Z'); title('XZ proj'); 
            subplot(133); imagesc(z_range, y_range, squeeze(max(pa_img2(:,:,:),[], 2)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'tickdir', 'out'); 
            ylabel('Y'); xlabel('Z'); title('YZ proj'); 
            subplot(132); imagesc(x_range, y_range, squeeze(max(pa_img2(:,:,:),[], 3)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; axis equal;
            ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out'); 
            % drawEllipsoidOverlay(Ellipse);%显示双声速范围
    
            filenameZX = sprintf('PAresult/Scan =%d zx.png',frame);
            filenameZY = sprintf('PAresult/Scan =%d zy.png',frame);
            filenameXY = sprintf('PAresult/Scan =%d xy.png',frame);
            imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],1))), filenameZX);
            imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],2))), filenameZY);
            imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],3))), filenameXY);

            disp(frame)

            % 开始计算 FWHM
            [max_val, max_idx] = max(pa_img2(:));
            [idx_x, idx_y, idx_z] = ind2sub(size(pa_img2), max_idx);
            
            % volumeViewer(pa_img2)
            
            directions = {'X', 'Y', 'Z'};
            half_max = max_val / 2;
            
            for dim = 1:3
                % --- A. 根据维度提取数据 ---
                if dim == 1
                    % X方向: 固定 Y 和 Z
                    axis_data = x_range(:); % 强制转为列向量
                    signal_data = double(squeeze(pa_img2(:, idx_y, idx_z)));
                    title_str = 'X-axis Profile';
                elseif dim == 2
                    % Y方向: 固定 X 和 Z
                    axis_data = y_range(:);
                    signal_data = double(squeeze(pa_img2(idx_x, :, idx_z)))'; 
                    % 注意: squeeze后通常是行向量，转置一下保统一
                    signal_data = signal_data(:); 
                    title_str = 'Y-axis Profile';
                else
                    % Z方向: 固定 X 和 Y
                    axis_data = z_range(:);
                    signal_data = double(squeeze(pa_img2(idx_x, idx_y, :)));
                    signal_data = signal_data(:);
                    title_str = 'Z-axis Profile';
                end
                
                % --- B. 计算 FWHM (由峰值向两侧搜索的稳健逻辑) ---
                % 1. 确定当前维度在 1D 剖面线中的主峰索引 (Center Index)
                if dim == 1
                    center_idx = idx_x;
                elseif dim == 2
                    center_idx = idx_y;
                else
                    center_idx = idx_z;
                end
                
                % 2. 向左搜索左边缘
                idx_L = center_idx;
                while idx_L > 1 && signal_data(idx_L) >= half_max
                    idx_L = idx_L - 1;
                end
                
                % 3. 向右搜索右边缘
                idx_R = center_idx;
                while idx_R < length(signal_data) && signal_data(idx_R) >= half_max
                    idx_R = idx_R + 1;
                end
                
                % 4. 检查是否成功找到边界 (防止信号在边界处仍未下降到半高以下)
                if signal_data(idx_L) >= half_max || signal_data(idx_R) >= half_max
                    fwhm_val = NaN; pos_L = NaN; pos_R = NaN;
                    warning('%s 方向信号未完整下降至半高以下，无法计算准确FWHM', directions{dim});
                else
                    % --- 左边缘插值 ---
                    % 此时 idx_L 是半高线外侧(低于半高)的点，idx_L+1 是半高线内侧(高于半高)的点
                    y_low_L = signal_data(idx_L);     y_high_L = signal_data(idx_L+1);
                    x_low_L = axis_data(idx_L);       x_high_L = axis_data(idx_L+1);
                    % 线性插值
                    pos_L = x_low_L + (half_max - y_low_L) * (x_high_L - x_low_L) / (y_high_L - y_low_L);
                    
                    % --- 右边缘插值 ---
                    % 此时 idx_R 是半高线外侧(低于半高)的点，idx_R-1 是半高线内侧(高于半高)的点
                    y_low_R = signal_data(idx_R);     y_high_R = signal_data(idx_R-1);
                    x_low_R = axis_data(idx_R);       x_high_R = axis_data(idx_R-1);
                    % 线性插值
                    pos_R = x_high_R + (half_max - y_high_R) * (x_low_R - x_high_R) / (y_low_R - y_high_R);
                    
                    % 计算宽度
                    fwhm_val = pos_R - pos_L;
                end
                fwhm_result(frame,dim) = fwhm_val;
            end
            disp(fwhm_result(frame,:))
        end

    case 5 %单声速循环
        for V_M = V_M_Range
            for frame = 1
                tic
                pa_data_frame = gpuArray(single(pa_data(:,:,frame))); % [Nelemt x Nsample]
                Points_sensor_all = gpuArray(single([x_sensor,y_sensor,z_sensor]));% [Nelemt x 3]
    
                % 调用 CUDA MEX 函数
                [pa_img, total_angle_weight] = SingleSpeedReconstraction_mex(Points_sensor_all, Points_img, pa_data_frame, single(fs), single(predelay), single(V_M), single(R));            % % 初始化图像缓冲区和角度权重
                toc

                % 收集GPU数据
                pa_img1 = gather(pa_img);
                total_angle_weight = gather(total_angle_weight);
                
                % 归一化并应用非线性增强
                pa_img2 = subplus(pa_img1)./total_angle_weight;

                % 显示不同视角的图像
                imin=1;
                imax = max(pa_img2,[],"all");
                k = 1;

                figure(4);        
                subplot(131); imagesc(z_range, x_range, squeeze(max(pa_img2(:,:,:),[],1)),[imin,imax*k]); 
                axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'tickdir', 'out'); 
                ylabel('X'); xlabel('Z'); title('XZ proj'); 
                subplot(133); imagesc(z_range, y_range, squeeze(max(pa_img2(:,:,:),[], 2)),[imin,imax*k]); 
                axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'tickdir', 'out'); 
                ylabel('Y'); xlabel('Z'); title('YZ proj'); 
                subplot(132); imagesc(x_range, y_range, squeeze(max(pa_img2(:,:,:),[], 3)),[imin,imax*k]); 
                axis equal tight; colormap gray; colorbar; axis equal;
                ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out'); 
                set(gca, 'tickdir', 'out');title(['VM = ',num2str(V_M)]); 
                
                filenameZX = sprintf('PAresult/Single Speed Loop zx VM=%.1f.png', V_M);
                filenameZY = sprintf('PAresult/Single Speed Loop zy VM=%.1f.png', V_M);
                filenameXY = sprintf('PAresult/Single Speed Loop xy VM=%.1f.png', V_M);
                imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],1))), filenameZX);
                imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],2))), filenameZY);
                imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],3))), filenameXY);
    
            end
        end

    case 6 %单声速多帧旋转复合
        pa_total = zeros(size(Points_img(:,:,:,1)),'single');
        pa_img_frames = zeros(Npixel_x,Npixel_y,Npixel_z,Nframe);

        if Is_Gating == 1
            [T, D, F] = size(pa_data(:,2501:3000,:));
            reshaped_data = reshape(pa_data(:,2501:3000,:), T*D, F);
            corr_mat = corr(reshaped_data);
            corr_line = mean(corr_mat,1);
            corr_line = corr_line/max(corr_line);%避免静止帧相关性太强导致归一化后系数偏小
            corr_line(1:10) = 0; %先采集后旋转时，去除未旋转的数据
            static_frames = 1:Nframe;
            top_vals = maxk(corr_line, 20);%找出最大的20个值，并按降序排列
            Similarity_threshold = top_vals(end);%动态调整复合时所用的相似度阈值，避免不同组数据相似度波动导致复合帧数不同
            static_frames = static_frames(corr_line>=Similarity_threshold);
            % 绘制相关系数图
            figure(11),plot(corr_line,'b'),hold on
            for isf = static_frames
                plot(isf,corr_line(isf),'*r'),hold on
            end
            hold off;
        else
            static_frames = 1:PA_FRAME_COMPOUND;
        end
        
        % 软件中触发速度11000对应0.800°，那么软件中触发速度10000可线性计算-改
        delta_angle = -11000*0.800/11000; %角度°
        static_Nframe = size(static_frames,2);

        firstframe_flag = 1; %首帧标识 1：是首帧，0：非首帧

        for frame = 1:1:static_Nframe %注意这里的step*delta_angle为相邻复合角
    
            tic
             % 坐标旋转(角度制,顺时针为负，逆时针为正 ：对应Y轴正方向朝上，否则相反）
            theta_x = 0; % 以x轴为中心旋转 
            theta_y = 0; % 以y轴为中心旋转 
            theta_z = (static_frames(frame)-1)*delta_angle; % 以z轴为中心旋转 
                
            %坐标整体平移(这里的坐标平移会影响椭球中心选取)
            trans_x = 0;
            trans_y = 0;
            trans_z = 0;
            
            rotate_x_mat = [1 0 0 0;0 cosd(theta_x) -sind(theta_x),0;0 sind(theta_x) cosd(theta_x) 0;0 0 0 1];
            rotate_y_mat = [cosd(theta_y) 0 -sind(theta_y) 0;0 1 0,0;sind(theta_y) 0 cosd(theta_y) 0;0 0 0 1];
            rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0;sind(theta_z) cosd(theta_z) 0,0;0 0 1 0;0 0 0 1];
            trans_mat = [1 0 0 trans_x;0 1 0 trans_y;0 0 1 trans_z;0 0 0 1];
            
            afine_mat = trans_mat*rotate_x_mat*rotate_y_mat*rotate_z_mat; %以原点为中心，先转z轴，再转y轴，再转x轴，最后平移
            detector_corr=detector_new*afine_mat'; 

            % 获取并调整传感器位置，Z 坐标取负以调整坐标系统
            x_sensor=detector_corr(:,1);
            y_sensor=detector_corr(:,2);
            z_sensor=-detector_corr(:,3); % 注意此处z轴坐标反转
            
            % 将传感器坐标移至GPU以加速计算
            x_sensor = gpuArray(single(x_sensor));
            y_sensor = gpuArray(single(y_sensor));
            z_sensor = gpuArray(single(z_sensor));

            pa_data_frame = gpuArray(single(pa_data(:,:,static_frames(frame)))); % [Nelemt x Nsample]
            Points_sensor_all = gpuArray(single([x_sensor,y_sensor,z_sensor]));% [Nelemt x 3]


            % pa_data_frame = gpuArray(single(pa_data(:,:,static_frames(frame)))); % [Nelemt x Nsample]
            % Points_sensor_all = gpuArray(single(detector_corr(:,1:3)));% [Nelemt x 3]
            tic
            % 调用 CUDA MEX 函数
            [pa_img, total_angle_weight] = SingleSpeedReconstraction_mex(Points_sensor_all, Points_img, pa_data_frame, single(fs), single(predelay), single(V_M), single(R));            % % 初始化图像缓冲区和角度权重
            toc
            disp(['frame : ',num2str(frame)]);
            
            % 收集GPU数据
            pa_img1 = gather(pa_img);
            total_angle_weight = gather(total_angle_weight);
            
            % 归一化并应用非线性增强
            pa_img2 = pa_img1./total_angle_weight;

            if firstframe_flag == 0 %判断是否为首帧
                %图像配准
                % pa_img2 = rigidRegistration3D(pa_ref, pa_img2);

            else
                pa_ref = pa_img2;%固定首帧作为参考帧
    
            end
            pa_total = pa_total+pa_img2;
            pa_img_frames(:,:,:,frame) = pa_img2;
            firstframe_flag = 0;%关闭首帧标识

            % 3D图像查看
            % volumeViewer(pa_img2);
            
            % 找到最大强度点的位置
            % [ym, xm, zm] = ind2sub(size(pa_img2), find(pa_img2 == max(pa_img2(:)))); 
            
            % 显示不同视角的图像
            imin=0;
            imax=max(pa_total,[],"all");
            
            figure(1); set (gca,'position',[0.1,0.1,0.8,0.8]);
            subplot(131); imagesc(z_range, x_range, squeeze(max(pa_total(:,:,:),[],1))); 
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'tickdir', 'out'); 
            ylabel('X'); xlabel('Z'); title('XZ proj'); 
            subplot(133); imagesc(z_range, y_range, squeeze(max(pa_total(:,:,:),[],2))); 
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'tickdir', 'out'); 
            ylabel('Y'); xlabel('Z'); title('YZ proj'); 
            subplot(132); imagesc(x_range, y_range, squeeze(max(pa_total(:,:,:),[],3))); 
            axis equal tight; colormap gray; colorbar; axis equal;
            ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out'); 
            title(frame)

            filenameZX = sprintf('PAresult/Single Speed Compounding zx frame=%d,V_M=%.1f.png',frame, V_M);
            filenameZY = sprintf('PAresult/Single Speed Compounding zy frame=%d,V_M=%.1f.png',frame, V_M);
            filenameXY = sprintf('PAresult/Single Speed Compounding xy frame=%d,V_M=%.1f.png',frame, V_M);
            imwrite(mat2gray(squeeze(max(pa_total(:,:,:),[],1))), filenameZX);
            imwrite(mat2gray(squeeze(max(pa_total(:,:,:),[],2))), filenameZY);
            imwrite(mat2gray(squeeze(max(pa_total(:,:,:),[],3))), filenameXY);
        end

    case 7 %双声速复合
        pa_total = zeros(size(Points_img(:,:,:,1)));
        pa_img_frames = zeros(Npixel_x,Npixel_y,Npixel_z,Nframe);

        if Is_Gating == 1
            sub_data = pa_data(:,2501:3000,:); % 提取目标区间数据
            [T, D, F] = size(sub_data);
            reshaped_data = reshape(sub_data, T*D, F);
    
            % 直接计算所有列之间的相关系数矩阵 (F x F)
            % MATLAB 的 corr 函数比在循环里调 corrcoef 快得多
            corr_matrix = corr(reshaped_data);
            corr_line = mean(corr_matrix, 1);
            corr_line = corr_line / max(corr_line(:));
    
            static_frames = 1:Nframe;
            top_vals = maxk(corr_line, 20);%找出最大的20个值，并按降序排列
            Similarity_threshold = top_vals(end);
            static_frames = static_frames(corr_line>=Similarity_threshold);
            % 绘制相关系数图
            figure(11),plot(corr_line,'b'),hold on
            for isf = static_frames
                plot(isf,corr_line(isf),'*r'),hold on
            end
            hold off;
        else
            static_frames = 1:Nframe;
        end
        
        % 软件中触发速度11000对应0.800°，那么软件中触发速度10000可线性计算-改
        delta_angle = -11000*0.800/11000; %角度°
        static_Nframe = size(static_frames,2);

        firstframe_flag = 1; %首帧标识 1：是首帧，0：非首帧

        for frame = 1:1:static_Nframe %注意这里的step*delta_angle为相邻复合角
            
            % 坐标旋转(角度制,顺时针为负，逆时针为正）
            theta_x = 0; % 以x轴为中心旋转 
            theta_y = 0; % 以y轴为中心旋转 
            theta_z = (static_frames(frame)-1)*delta_angle; % 以z轴为中心旋转 
            
            %坐标平移(这里的坐标平移会影响椭球中心选取)
            trans_x = 0;
            trans_y = 0;
            trans_z = 0;
            
            rotate_x_mat = [1 0 0 0;0 cosd(theta_x) -sind(theta_x),0;0 sind(theta_x) cosd(theta_x) 0;0 0 0 1];
            rotate_y_mat = [cosd(theta_y) 0 -sind(theta_y) 0;0 1 0,0;sind(theta_y) 0 cosd(theta_y) 0;0 0 0 1];
            rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0;sind(theta_z) cosd(theta_z) 0,0;0 0 1 0;0 0 0 1];
            trans_mat = [1 0 0 trans_x;0 1 0 trans_y;0 0 1 trans_z;0 0 0 1];
            afine_mat = trans_mat*rotate_x_mat*rotate_y_mat*rotate_z_mat; %以原点为中心，先转z轴，再转y轴，再转x轴，最后平移
            
            detector_corr = detector_new*afine_mat'; 
            % 获取并调整传感器位置，Z 坐标取负以调整坐标系统
            x_sensor=detector_corr(:,1);
            y_sensor=detector_corr(:,2);
            z_sensor=-detector_corr(:,3); % 注意此处z轴坐标反转
            
            % 将传感器坐标移至GPU以加速计算
            x_sensor = gpuArray(single(x_sensor));
            y_sensor = gpuArray(single(y_sensor));
            z_sensor = gpuArray(single(z_sensor));

            pa_data_frame = gpuArray(single(pa_data(:,:,static_frames(frame)))); % [Nelemt x Nsample]
            Points_sensor_all = gpuArray(single([x_sensor,y_sensor,z_sensor]));% [Nelemt x 3]

            % 调用 CUDA MEX 函数
            [pa_img, total_angle_weight] = DualSpeedReconstraction_mex([Ellipse.a,Ellipse.b,Ellipse.c,Ellipse.centerx,Ellipse.centery,Ellipse.centerz], ...
                                                                Points_sensor_all, Points_img, pa_data_frame, ...
                                                                single(fs), single(predelay), single(VM_out), single(VM_in), single(R));

            disp(['frame : ',num2str(frame)]);
            
            % 收集GPU数据
            pa_img1 = gather(pa_img);
            total_angle_weight = gather(total_angle_weight);
            
            % 归一化并应用非线性增强
            pa_img2 = pa_img1./total_angle_weight;

            % if firstframe_flag == 0 %判断是否为首帧
            %     %图像配准
            %     pa_img2 = rigidRegistration3D(pa_ref, pa_img2);
            % 
            % else
            %     pa_ref = pa_img2;%固定首帧作为参考帧
            % 
            % end
            pa_total = pa_total+pa_img2;
            pa_img_frames(:,:,:,frame) = pa_img2;

            firstframe_flag = 0;%关闭首帧标识

            % 显示不同视角的图像
            imin=0;
            imax=max(pa_total,[],"all");
            
            figure(7); set (gca,'position',[0.1,0.1,0.8,0.8]);
            subplot(131); imagesc(z_range, x_range, squeeze(max(pa_total(:,:,:),[],1)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');set(gca, 'tickdir', 'out'); 
            ylabel('X'); xlabel('Z'); title('XZ proj'); 
            subplot(133); imagesc(z_range, y_range, squeeze(max(pa_total(:,:,:),[], 2)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');set(gca, 'tickdir', 'out'); 
            ylabel('Y'); xlabel('Z'); title('YZ proj'); 
            subplot(132); imagesc(x_range, y_range, squeeze(max(pa_total(:,:,:),[], 3)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');
            ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out'); 
            drawEllipsoidOverlay(Ellipse);%显示双声速范围

            filenameZX = sprintf('PAresult/Dual Speed Compounding zx frame=%d,VM_out=%.1f,VM_in=%.1f.png',frame, VM_out,VM_in);
            filenameZY = sprintf('PAresult/Dual Speed Compounding zy frame=%d,VM_out=%.1f,VM_in=%.1f.png',frame, VM_out,VM_in);
            filenameXY = sprintf('PAresult/Dual Speed Compounding xy frame=%d,VM_out=%.1f,VM_in=%.1f.png',frame, VM_out,VM_in);
            imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],1))), filenameZX);
            imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],2))), filenameZY);
            imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],3))), filenameXY);
            
        end
        

    case 8 %单声速相干因子法多帧复合
        pa_total = zeros(size(Points_img(:,:,:,1)));

        % 软件中触发速度11000对应0.800°，那么软件中触发速度5000可线性计算
        delta_angle = -10000*0.800/11000; %角度°

        % 坐标旋转(角度制,顺时针为负，逆时针为正）
        theta_x = 0; % 以x轴为中心旋转 
        theta_y = 0; % 以y轴为中心旋转 
        theta_z = delta_angle; % 以z轴为中心旋转 
        
        %坐标平移(这里的坐标平移会影响椭球中心选取)
        trans_x = 0;
        trans_y = 0;
        trans_z = 0;
        
        rotate_x_mat = [1 0 0 0;0 cosd(theta_x) -sind(theta_x),0;0 sind(theta_x) cosd(theta_x) 0;0 0 0 1];
        rotate_y_mat = [cosd(theta_y) 0 -sind(theta_y) 0;0 1 0,0;sind(theta_y) 0 cosd(theta_y) 0;0 0 0 1];
        rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0;sind(theta_z) cosd(theta_z) 0,0;0 0 1 0;0 0 0 1];
        trans_mat = [1 0 0 trans_x;0 1 0 trans_y;0 0 1 trans_z;0 0 0 1];
        afine_mat = trans_mat*rotate_x_mat*rotate_y_mat*rotate_z_mat; %以原点为中心，先转z轴，再转y轴，再转x轴，最后平移

        for frame = 1:Nframe
            
            detector_new=detector_new*afine_mat'; 

            % 获取并调整传感器位置，Z 坐标取负以调整坐标系统
            x_sensor=detector_new(:,1);
            y_sensor=detector_new(:,2);
            z_sensor=-detector_new(:,3); % 注意此处z轴坐标反转
            
            % 将传感器坐标移至GPU以加速计算
            x_sensor = gpuArray(single(x_sensor));
            y_sensor = gpuArray(single(y_sensor));
            z_sensor = gpuArray(single(z_sensor));

            pa_data_frame = gpuArray(single(pa_data(:,:,frame))); % [Nelemt x Nsample]
            Points_sensor_all = gpuArray(single([x_sensor,y_sensor,z_sensor]));% [Nelemt x 3]

            % 调用 CUDA MEX 函数
            tic
            [pa_img, total_angle_weight, coherent_factor, ~] = SingleSpeedReconstraction_cof_mex(Points_sensor_all, Points_img, pa_data_frame, single(fs), single(predelay), single(V_M), single(R));            % % 初始化图像缓冲区和角度权重
            toc
            disp(['frame : ',num2str(frame)]);
            
            % 收集GPU数据
            pa_img1 = gather(pa_img);
            total_angle_weight = gather(total_angle_weight);
            coherent_factor = gather(coherent_factor);
            
            % 归一化并应用非线性增强
            pa_img2 = pa_img1.*coherent_factor./total_angle_weight;

            if frame>1
            %图像配准
            % pa_img2 = rigidRegistration3D(pa_total, pa_img2);

            end
            pa_total = pa_total+pa_img2;

            % 3D图像查看
            % volumeViewer(pa_img2);
            
            % 找到最大强度点的位置
            % [ym, xm, zm] = ind2sub(size(pa_img2), find(pa_img2 == max(pa_img2(:)))); 
            
            % 显示不同视角的图像
            imin=0;
            imax=max(pa_total,[],"all");
            
            figure(1); set (gca,'position',[0.1,0.1,0.8,0.8]);
            subplot(131); imagesc(z_range, x_range, squeeze(max(pa_total(:,:,:),[],1)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');set(gca, 'tickdir', 'out'); 
            ylabel('X'); xlabel('Z'); title('XZ proj'); 
            subplot(133); imagesc(z_range, y_range, squeeze(max(pa_total(:,:,:),[], 2)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');set(gca, 'tickdir', 'out'); 
            ylabel('Y'); xlabel('Z'); title('YZ proj'); 
            subplot(132); imagesc(x_range, y_range, squeeze(max(pa_total(:,:,:),[], 3)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');
            ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out'); 

            filenameZX = sprintf('zx frame=%d,V_M=%.1f.png',frame, V_M);
            filenameZY = sprintf('zy frame=%d,V_M=%.1f.png',frame, V_M);
            filenameXY = sprintf('xy frame=%d,V_M=%.1f.png',frame, V_M);
            imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],1))), filenameZX);
            imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],2))), filenameZY);
            imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],3))), filenameXY);

        end

    case 9 %单/双声速旋转平移复合
        %% 子程序对应采集数据说明：
        % 使用的双声速重建的函数，但双声速功能尚未修改完成
        % 为了保证不同数据的方向相同，程序固定第一帧为复合模板，因此在采集数据时应当先采集再旋转
        % 当前程序设定采集帧率为10Hz，程序中包含未旋转帧过滤程序，默认过滤从开始采集后的部分数据，确保复合数据均为旋转数据
        
        imgsize = size(Points_img,1:3);
        pa_img_total = zeros(imgsize+[step_length_y*resolution_factor*(Nframey_scan-mod(Nframey_scan-1,step_y)-1) ...
                                           step_length_x*resolution_factor*(Nframex_scan-mod(Nframex_scan-1,step_x)-1) 0]);    
        [totalsize_y,totalsize_x,totalsize_z] = size(pa_img_total);%最终平移拼接后的图像像素尺寸，单位像素数
        pa_count_total = zeros(totalsize_y, totalsize_x, totalsize_z, 'single');%计数矩阵，用于计算掩膜权重和帧数的影响
        x_range_total = -totalsize_x/resolution_factor/2:totalsize_x/resolution_factor/2;%物理长度坐标，单位mm
        y_range_total = -totalsize_y/resolution_factor/2:totalsize_y/resolution_factor/2;
        z_range_total = -totalsize_z/resolution_factor/2:totalsize_z/resolution_factor/2 + center_z;

        %% --- 绘图窗口初始化 ---
        f9 = figure(9); 
        subplot(131); 
        h_img9_1 = imagesc(x_range, y_range, zeros(length(y_range), length(x_range))); % 初始化空图，拿到句柄 h_img9_1
        axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');
        ylabel('Y'); xlabel('X'); title('pa img XY proj'); set(gca, 'tickdir', 'out');
        
        subplot(132); 
        h_img9_2 = imagesc(x_range, y_range, zeros(length(y_range), length(x_range))); % 拿到句柄 h_img9_2
        axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');
        ylabel('Y'); xlabel('X'); title('GaussianMask XY proj'); set(gca, 'tickdir', 'out');
        
        subplot(133); 
        h_img9_3 = imagesc(x_range, y_range, zeros(length(y_range), length(x_range))); % 拿到句柄 h_img9_3
        axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');
        ylabel('Y'); xlabel('X'); title('pa img masked XY proj'); set(gca, 'tickdir', 'out');
        
        % --- 在循环外初始化 Figure 10 (同理) ---
        f10 = figure(10);
        subplot(131); h_img10_1 = imagesc(z_range_total, x_range_total, zeros(length(x_range_total), length(z_range_total))); % 注意尺寸
        axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');set(gca, 'tickdir', 'out');
        ylabel('X'); xlabel('Z');title('ZX proj'); 
        subplot(132); h_img10_2 = imagesc(x_range_total, y_range_total, zeros(length(y_range_total), length(x_range_total))); % 注意尺寸
        axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');set(gca, 'tickdir', 'out');
        ylabel('Y'); xlabel('X');title('XY proj'); 
        subplot(133); h_img10_3 = imagesc(z_range_total, y_range_total, zeros(length(y_range_total), length(z_range_total))); % 注意尺寸
        axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');set(gca, 'tickdir', 'out');
        ylabel('Y'); xlabel('Z');title('ZY proj'); 

        VM_out_start = VM_out;
        
        for xframe = 1:step_x:Nframex_scan

            for yframe = 1:step_y:Nframey_scan %文件数量方向

                VM_out = VM_out_start-0.09*(yframe-1 + yframe*(xframe-1)); %水温降低的声速补偿，若实验使用冷水则关闭该补偿
                VM_in = VM_out;
                frame_idx = 1+((xframe-1)*Nframey_scan+(yframe-1))*4;
                [datax,DAQ_time_point] = func_3D_PACT_Data_Time_Read(folder_path,str_name(frame_idx).name);
                
                % 根据表面信号判断区分超声帧和光声帧
                frame1_val = max(sum(datax(:, 1:100, 1)));
                frame2_val = max(sum(datax(:, 1:100, 2)));
                offset = (frame1_val < frame2_val); 
                pa_idx = (1 + offset) : 2 : size(datax, 3); % 光声帧索引
                us_idx = (2 - offset) : 2 : size(datax, 3); % 超声帧索引
                
                pa_data = -datax(:,:,pa_idx);%选择光声帧数
                figure(1);imagesc(pa_data(:,:,1),[-100,100]);
                if Is_Denoising == 1 
                    pa_data = denoise_sinogram(pa_data);%滤除换能器带宽外的噪声
                end

                pa_data_frame = gpuArray(single(pa_data(:,:,yframe))); % [Nelemt x Nsample]
                detector_new = gpuArray(single([x_sensor,y_sensor,z_sensor,z_sensor*0+1]));% [Nelemt x 3]
        %
                pa_total = zeros(size(Points_img(:,:,:,1)),'single');
        
                tic
                if Is_Gating == 1
                    [T, D, F] = size(pa_data(:,2501:3000,:));% 提取目标区间数据
                    reshaped_data = reshape(pa_data(:,2501:3000,:), T*D, F);
                    corr_mat = corr(reshaped_data);
                    corr_line = mean(corr_mat,1);
                    corr_line = corr_line/max(corr_line);%避免静止帧相关性太强导致归一化后系数偏小
                    corr_line(1:10) = 0; %先采集后旋转时，去除未旋转的数据
                    static_frames = 1:Nframe;
                    top_vals = maxk(corr_line, 20);%找出最大的20个值，并按降序排列
                    Similarity_threshold = top_vals(end);%动态调整复合时所用的相似度阈值，避免不同组数据相似度波动导致复合帧数不同
                    static_frames = static_frames(corr_line>=Similarity_threshold);
                    % 绘制相关系数图
                    figure(11),plot(corr_line,'b'),hold on
                    for isf = static_frames
                        plot(isf,corr_line(isf),'*r'),hold on
                    end
                    hold off;
                else
                    static_frames = 1:Nframe;
                end
                
                % 软件中触发速度11000对应0.800°，那么软件中触发速度10000可线性计算-改
                delta_angle = -11000*0.800/11000; %角度°
                static_Nframe = size(static_frames,2);

                firstframe_flag = 1; %首帧标识 1：是首帧，0：非首帧

                for frame = 1:1:static_Nframe %注意这里的step*delta_angle为相邻复合角
            
                    tic
                     % 坐标旋转(角度制,顺时针为负，逆时针为正 ：对应Y轴正方向朝上，否则相反）
                    theta_x = 0; % 以x轴为中心旋转 
                    theta_y = 0; % 以y轴为中心旋转 
                    theta_z = (static_frames(frame)-1)*delta_angle; % 以z轴为中心旋转 
                        
                    %坐标整体平移(这里的坐标平移会影响椭球中心选取)
                    trans_x = 0;
                    trans_y = 0;
                    trans_z = 0;
                    
                    rotate_x_mat = [1 0 0 0;0 cosd(theta_x) -sind(theta_x),0;0 sind(theta_x) cosd(theta_x) 0;0 0 0 1];
                    rotate_y_mat = [cosd(theta_y) 0 -sind(theta_y) 0;0 1 0,0;sind(theta_y) 0 cosd(theta_y) 0;0 0 0 1];
                    rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0;sind(theta_z) cosd(theta_z) 0,0;0 0 1 0;0 0 0 1];
                    trans_mat = [1 0 0 trans_x;0 1 0 trans_y;0 0 1 trans_z;0 0 0 1];
                    
                    afine_mat = trans_mat*rotate_x_mat*rotate_y_mat*rotate_z_mat; %以原点为中心，先转z轴，再转y轴，再转x轴，最后平移
                    detector_corr = detector_new*afine_mat'; 
        
                    pa_data_frame = gpuArray(single(pa_data(:,:,static_frames(frame)))); % [Nelemt x Nsample]
                    Points_sensor_all = gpuArray(single(detector_corr(:,1:3)));% [Nelemt x 3]
                    
                    % 调用 CUDA MEX 函数
                    [pa_img, total_angle_weight] = DualSpeedReconstraction_mex([Ellipse.a,Ellipse.b,Ellipse.c,Ellipse.centerx,Ellipse.centery,Ellipse.centerz], ...
                                                                        Points_sensor_all, Points_img, pa_data_frame, ...
                                                                        single(fs), single(predelay), single(VM_out), single(VM_in), single(R));
                     
                    disp(['xframe : ',num2str(xframe),' yframe : ',num2str(yframe),' frame : ',num2str(frame)]);
                    
                    % 收集GPU数据
                    pa_img1 = gather(pa_img);
                    total_angle_weight = gather(total_angle_weight);
                    
                    % 归一化并应用非线性增强
                    pa_img2 = pa_img1./total_angle_weight;
                    [GaussianMask, ~, ~] = generateGaussianMask({x_range, y_range}, 'Center', [center_x,center_y], 'Sigma', GaussianMask_FWHM*0.425);
                    GaussianMask = GaussianMask.*ones(1,1,size(z_range,2));%z方向均匀
                    pa_img3 = pa_img2.*GaussianMask;
                    % pa_img3 = pa_img3/max(pa_img3,[],'all');
                    % pa_img3 = pa_img2.*(2-GaussianMask); %取消高斯滤波并进行高斯增强
        
                    if firstframe_flag == 0 %判断是否为首帧
                        %图像配准
                        % pa_img3 = rigidRegistration3D(pa_ref, pa_img3);
                    else
                        pa_ref = pa_img3;%固定首帧作为参考帧
                    end

                    pa_total = pa_total+subplus(pa_img3);
                    firstframe_flag = 0;%关闭首帧标识
                    toc
        
                    % 显示不同视角的图像
                    imin=0;
                    imax=max(pa_total,[],"all");
                    % --- 通过句柄更新数据，不会抢焦点 ---
                    set(h_img9_1, 'CData', squeeze(max(pa_img2(:,:,:),[], 3)));
                    set(h_img9_2, 'CData', squeeze(max(GaussianMask(:,:,:),[], 3)));
                    set(h_img9_3, 'CData', squeeze(max(pa_total(:,:,:),[], 3)));
                                     
                    drawnow limitrate;
                    % filenameZX = sprintf('zx xframe=%d, yframe=%d, frame=%d.png', xframe, yframe, frame);
                    % filenameZY = sprintf('zy xframe=%d, yframe=%d, frame=%d.png', xframe, yframe, frame);
                    % filenameXY = sprintf('xy xframe=%d, yframe=%d, frame=%d.png', xframe, yframe, frame);
                    % imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],1))), filenameZX);
                    % imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],2))), filenameZY);
                    % imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],3))), filenameXY);
                end % frame end

                toc
                SubWinSize_y = imgsize(1)-1;
                SubWinSize_x = imgsize(2)-1;
                SubWinSize_z = imgsize(3)-1;

                idx_y = totalsize_y-(yframe-1)*step_length_y*resolution_factor-SubWinSize_y...
                       :totalsize_y-(yframe-1)*step_length_y*resolution_factor;
                idx_x = (xframe-1)*step_length_x*resolution_factor+1 ...
                       :(xframe-1)*step_length_x*resolution_factor+1+SubWinSize_x;

                current_weight = GaussianMask * single(static_Nframe);%单组数据权重

                pa_count_total(idx_y,idx_x,:) = pa_count_total(idx_y,idx_x,:) + current_weight;%计算每帧图像累加贡献的权重矩阵

                pa_img_total(idx_y,idx_x,:) = pa_img_total(idx_y,idx_x,:) + pa_total;
                pa_img_total_2 = pa_img_total ./ (pa_count_total + eps);%去除帧数不均对亮度的影响，避免除以0
                imin=min(pa_img_total_2,[],"all");
                imax=max(pa_img_total_2,[],"all");

                % 更新 Figure 10 的数据
                set(h_img10_1, 'CData', squeeze(max(pa_img_total_2(:,:,:),[],1)));
                set(h_img10_1.Parent, 'CLim', [imin, imax]); 
                set(h_img10_2, 'CData', squeeze(max(pa_img_total_2(:,:,:),[],3)));
                set(h_img10_2.Parent, 'CLim', [imin, imax]); 
                set(h_img10_3, 'CData', squeeze(max(pa_img_total_2(:,:,:),[],2)));
                set(h_img10_3.Parent, 'CLim', [imin, imax]); 
                % drawEllipsoidOverlay(Ellipse);%显示双声速范围
                
                drawnow limitrate;
                            
                filenameZX = sprintf('PAresult/scan step=%d zx xframe=%d, yframe=%d.png',step_x, xframe, yframe);
                filenameZY = sprintf('PAresult/scan step=%d zy xframe=%d, yframe=%d.png',step_x, xframe, yframe);
                filenameXY = sprintf('PAresult/scan step=%d xy xframe=%d, yframe=%d.png',step_x, xframe, yframe);
                imwrite(mat2gray(squeeze(max(pa_img_total_2(end:-1:1,:,:),[],1))), filenameZX);
                imwrite(mat2gray(squeeze(max(pa_img_total_2(end:-1:1,:,:),[],2))), filenameZY);
                imwrite(mat2gray(squeeze(max(pa_img_total_2(end:-1:1,:,:),[],3))), filenameXY);
            end %yframe end
        end %xframe end
            pa_img_total_2_cut = pa_img_total_2(1:end-150,1:end-100,140:end-60);%切掉图像多余部分，需确认参数
            % volumeViewer(pa_img_total_2_cut);
            save('pa_img_total_step1.mat', 'pa_img_total_2', '-v7.3');

       case 10  % 超声旋转复合
            [num_spl, num_rcv, num_rot] = size(dataUS); 
            iq_ch = reshape(hilbert(reshape(dataUS, num_spl, [])), [num_spl, num_rcv, num_rot]);

            fs = 40e6;
            fc = 3.5e6;
            
            c = V_US;
            ang_hole = 16.5/180*pi;
            f_xdc = 15e-3;
            rad_xdc = 108.85e-3;
            
            del_tx = f_xdc/c;
            rad_src = rad_xdc - f_xdc;
            wvl = c/fc;
            x_src = - rad_src * sin(ang_hole);
            y_src = 0;
            z_src = rad_src * cos(ang_hole);
            src = [ x_src, y_src, -z_src ];
            
            rcv = detector * 1e-3;
            rcv(:,3) = -rcv(:,3);
            
            % acq/beamforming specs
            no_rcv = 1:1024;
            del_acq = 100e-6;
            t0 = del_acq;
            
            % mechanical specs
            no_pos = 1:1:100;
            rng_rot = -80; % for crossed hair
            a_rot = 0:-0.8:99*-0.811;
            num_pos = numel(no_pos);

            % imaging volume
            dp = 1e-3 / resolution_factor;
            x_sp = [x_range(1), x_range(end)] * 1e-3; % sp: span
            y_sp = [y_range(1), y_range(end)] * 1e-3;
            z_sp = [z_range(1), z_range(end)] * 1e-3;
                
            x_1d = x_range * 1e-3;
            y_1d = y_range * 1e-3;
            z_1d = z_range * 1e-3;

            num_xp = numel( x_1d );
            num_yp = numel( y_1d );
            num_zp = numel( z_1d );
            [ x_2d , y_2d ] = ndgrid( x_1d , y_1d );
            num_pnt_xy = numel(x_2d);
            
            % beamforming
            % --- 修改 beamforming 部分 ---
            beta = 10/180*pi; 
            fd = 0;
            das_params = [t0, fs, fd, c, beta, del_tx];
            
            corr_ang_x = theta_x;   % 绕X轴旋转 (上下倾斜)
            corr_ang_y = theta_y;   % 绕Y轴旋转 (左右倾斜)
            corr_ang_z = theta_z;   % 绕Z轴旋转 (平面内旋转)
           
            Rx = [1 0 0; 0 cosd(corr_ang_x) -sind(corr_ang_x); 0 sind(corr_ang_x) cosd(corr_ang_x)];
            Ry = [cosd(corr_ang_y) 0 -sind(corr_ang_y); 0 1 0; sind(corr_ang_y) 0 cosd(corr_ang_y)];
            Rz = [cosd(corr_ang_z) -sind(corr_ang_z) 0; sind(corr_ang_z) cosd(corr_ang_z) 0; 0 0 1];
            R_corr = Rz * Ry * Rx; % 全局校正矩阵
            
            if Is_Gating==1
                sub_data = dataUS(:, :, :); 
                [T, D, F] = size(sub_data);
                reshaped_data = reshape(sub_data, T*D, F);
                
                corr_matrix = corr(reshaped_data);
                
                corr_line = mean(corr_matrix, 1);
                corr_line = corr_line / max(corr_line(:));
                
                static_frames = 1:num_rot;
                top_vals = maxk(corr_line, US_FRAME_COMPOUND);
                Similarity_threshold = top_vals(end);
                static_frames = static_frames(corr_line>=Similarity_threshold);
            else
                static_frames = 1:US_FRAME_COMPOUND;
            end
            
            % % --- 绘图展示静态帧 ---
            % figure;
            % plot(corr_line, 'b'); hold on;
            % % 使用一次 plot 绘制所有散点，比在循环里 plot 快得多
            % if ~isempty(static_frames)
            %     plot(static_frames, corr_line(static_frames), 'r*');
            % end
            % hold off;
            % title('Signal Correlation Analysis');
            % toc
            
            iq_im_sum = zeros(num_xp, num_yp, num_zp);
            % iq_image_frame = zeros(num_xp, num_yp, num_zp,num_pos);
            
            for ii = 1 : num_rot
                if ~ismember(ii,static_frames)
                    continue
                end
                tic;
                % 1. 旋转计算 (保持原样)
                no_pos_i = no_pos(ii);
                a_i = a_rot(no_pos_i);
                R_i = rotz(a_i);
                src_i = (R_i * src')';
                rcv_i = (R_i * rcv(no_rcv, :)')';
            
                src_i = (R_corr * src_i')';
                rcv_i = (R_corr * rcv_i')';
                
                % 2. Update Orientation (保持原样)
                ori_rcv_i = [0, 0, 0] - rcv_i;
                ori_rcv_i = ori_rcv_i ./ vecnorm(ori_rcv_i, 2, 2);
                
                % 3. 调用 CUDA 加速重建 (一行代码替代原来的 jj 循环)
                % 注意：x_1d, y_1d, z_1d 是向量
                iq_im_pos_i = mex_das_gpu(iq_ch(:, no_rcv, no_pos_i), ...
                                          x_1d, y_1d, z_1d, ...
                                          rcv_i, src_i, ori_rcv_i, ...
                                          das_params);
                
                % 4. 累加结果
                % CUDA 输出已经是 (num_xp, num_yp, num_zp) 格式
                % 如果你后面需要保存单帧结果，直接保存 iq_im_pos_i
                iq_im_sum = iq_im_sum + iq_im_pos_i;
                % iq_image_frame(:,:,:,ii) = iq_im_pos_i;

                dr = Dynamic_Range;
                
                bm_im = abs(iq_im_sum);
                bm_im = bm_im / max(bm_im, [], 'all');
                bm = 20*log10(bm_im);
                bm(bm<-dr) = -dr;
                bm_total = bm +dr;

                filenameZX = sprintf('USresult/US Compounding zx frame=%d,sos=%.1f.png',ii, c);
                filenameZY = sprintf('USresult/US Compounding zy frame=%d,sos=%.1f.png',ii, c);
                filenameXY = sprintf('USresult/US Compounding xy frame=%d,sos=%.1f.png',ii, c);
                imwrite(mat2gray(squeeze(max(bm_total(:,:,:),[],2))), filenameZX);
                imwrite(mat2gray(squeeze(max(bm_total(:,:,:),[],1))), filenameZY);
                imwrite(mat2gray(squeeze(max(bm_total(:,:,:),[],3))'), filenameXY);
                
                fprintf(1, 'CUDA 3D Recon for Pos %d finished. %.4f sec used.\n', ii, toc);
            end

            dr = Dynamic_Range;
            bm_im = abs(iq_im_sum);
            bm_im = bm_im / max(bm_im, [], 'all');
            bm = 20*log10(bm_im);
            bm(bm<-dr) = -dr;
            bm = bm +dr;
    
            figure();
            subplot(131); imagesc(z_sp*1e3, x_sp*1e3, squeeze(max(bm(:,:,:),[],2))); 
            axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
            ylabel('X'); xlabel('Z'); title('XZ proj'); 
            subplot(133); imagesc(z_sp*1e3, y_sp*1e3, squeeze(max(bm(:,:,:),[], 1))); 
            axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
            ylabel('Y'); xlabel('Z'); title('YZ proj'); 
            subplot(132); imagesc(x_sp*1e3, y_sp*1e3, squeeze(max(bm(:,:,:),[], 3))'); 
            axis equal tight; colormap gray; colorbar; 
            ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out'); 


       case 11 % 超声声速遍历
            [num_spl, num_rcv, num_rot] = size(dataUS); 
            iq_ch = reshape(hilbert(reshape(dataUS, num_spl, [])), [num_spl, num_rcv, num_rot]);
            %
            fs = 40e6;
            fc = 3.5e6;
            
            ang_hole = 16.5/180*pi;
            f_xdc = 15e-3;
            rad_xdc = 108.85e-3;
            

            rad_src = rad_xdc - f_xdc;
            x_src = - rad_src * sin(ang_hole);
            y_src = 0;
            z_src = rad_src * cos(ang_hole);
            src = [ x_src, y_src, -z_src ];
            
            rcv = detector * 1e-3;
            rcv(:,3) = -rcv(:,3);
            
            % acq/beamforming specs
            no_rcv = 1:1024;
            del_acq = 100e-6;
            t0 = del_acq;
            
            % mechanical specs
            no_pos = 1:1:100;
            rng_rot = -80; % for crossed hair
            % a_rot = linspace(0, rng_rot, num_rot);
            a_rot = 0:-0.8:99*-0.811;
            num_pos = numel(no_pos);
            
            % imaging volume
            dp = 1e-3 / resolution_factor;
            x_sp = [x_range(1), x_range(end)] * 1e-3; % sp: span
            y_sp = [y_range(1), y_range(end)] * 1e-3;
            z_sp = [z_range(1), z_range(end)] * 1e-3;
                
            x_1d = x_range * 1e-3;
            y_1d = y_range * 1e-3;
            z_1d = z_range * 1e-3;

            num_xp = numel( x_1d );
            num_yp = numel( y_1d );
            num_zp = numel( z_1d );
            [ x_2d , y_2d ] = ndgrid( x_1d , y_1d );
            num_pnt_xy = numel(x_2d);
            
            % beamforming
            % --- 修改 beamforming 部分 ---
            beta = 10/180*pi; 
            fd = 0;
            % 参数打包: [t0, fs, fd, c, beta, del_tx]
                        
            corr_ang_x = theta_x;   % 绕X轴旋转 (上下倾斜)
            corr_ang_y = theta_y;   % 绕Y轴旋转 (左右倾斜)
            corr_ang_z = theta_z;   % 绕Z轴旋转 (平面内旋转)

            Rx = [1 0 0; 0 cosd(corr_ang_x) -sind(corr_ang_x); 0 sind(corr_ang_x) cosd(corr_ang_x)];
            Ry = [cosd(corr_ang_y) 0 -sind(corr_ang_y); 0 1 0; sind(corr_ang_y) 0 cosd(corr_ang_y)];
            Rz = [cosd(corr_ang_z) -sind(corr_ang_z) 0; sind(corr_ang_z) cosd(corr_ang_z) 0; 0 0 1];
            R_corr = Rz * Ry * Rx; % 全局校正矩阵

            for sos = V_M_Range
                c = sos;
                del_tx = f_xdc/c;
                das_params = [t0, fs, fd, c, beta, del_tx];
                
                
                iq_im_sum = zeros(num_xp, num_yp, num_zp);
    
                for ii = 1 : US_FRAME_COMPOUND
                    tic;
                    % 1. 旋转计算 (保持原样)
                    no_pos_i = no_pos(ii);
                    a_i = a_rot(no_pos_i);
                    R_i = rotz(a_i);
                    src_i = (R_i * src')';
                    rcv_i = (R_i * rcv(no_rcv, :)')';
                    src_i = (R_corr * src_i')';
                    rcv_i = (R_corr * rcv_i')';
                    
                    % 2. Update Orientation (保持原样)
                    ori_rcv_i = [0, 0, 0] - rcv_i;
                    ori_rcv_i = ori_rcv_i ./ vecnorm(ori_rcv_i, 2, 2);
                    
                    % 3. 调用 CUDA 加速重建 (一行代码替代原来的 jj 循环)
                    % 注意：x_1d, y_1d, z_1d 是向量
                    iq_im_pos_i = mex_das_gpu(iq_ch(:, no_rcv, no_pos_i), ...
                                              x_1d, y_1d, z_1d, ...
                                              rcv_i, src_i, ori_rcv_i, ...
                                              das_params);
                    
                    % 4. 累加结果
                    % CUDA 输出已经是 (num_xp, num_yp, num_zp) 格式
                    % 如果你后面需要保存单帧结果，直接保存 iq_im_pos_i
                    iq_im_sum = iq_im_sum + iq_im_pos_i;
    
                    % 5. 保存 (可选)
                    fprintf(1, 'CUDA 3D Recon for Pos %d finished. %.4f sec used.\n', ii, toc);
                end
                
                dr = Dynamic_Range;
                bm_im = abs(iq_im_sum);
                bm_im = bm_im / max(bm_im, [], 'all');
                bm = 20*log10(bm_im);
                bm(bm<-dr) = -dr;
                bm_total = bm +dr;
    
                filenameZX = sprintf('USresult/US SOS LOOP zx sos=%.1f.png', c);
                filenameZY = sprintf('USresult/US SOS LOOP zy sos=%.1f.png', c);
                filenameXY = sprintf('USresult/US SOS LOOP xy sos=%.1f.png', c);
                imwrite(mat2gray(squeeze(max(bm_total(:,:,:),[],2))), filenameZX);
                imwrite(mat2gray(squeeze(max(bm_total(:,:,:),[],1))), filenameZY);
                imwrite(mat2gray(squeeze(max(bm_total(:,:,:),[],3))'), filenameXY);
                
                figure(3);
                subplot(131); imagesc(z_sp, x_sp, squeeze(max(bm_total(:,:,:),[],2))); 
                axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
                ylabel('X'); xlabel('Z'); title('XZ proj'); 
                subplot(133); imagesc(z_sp, y_sp, squeeze(max(bm_total(:,:,:),[], 1))); 
                axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
                ylabel('Y'); xlabel('Z'); title('YZ proj'); 
                subplot(132); imagesc(x_sp, y_sp, squeeze(max(bm_total(:,:,:),[], 3))'); 
                axis equal tight; colormap gray; colorbar; 
                ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out'); 

                disp(sos)
            end

        case 12 % 超声扫描
        fwhm_result = zeros(289,3);
        for frame = 145:145
            [datax,DAQ_time_point] = func_3D_PACT_Data_Time_Read(folder_path,str_name(1 + (frame-1)*4).name);
            
            % 根据表面信号判断区分超声帧和光声帧
            frame1_val = max(sum(datax(:, 1:100, 1)));
            frame2_val = max(sum(datax(:, 1:100, 2)));
            offset = (frame1_val < frame2_val); 
            pa_idx = (1 + offset) : 2 : size(datax, 3); % 光声帧索引
            us_idx = (2 - offset) : 2 : size(datax, 3); % 超声帧索引
            
            data = datax(:,:,pa_idx);%选择光声帧数
            dataUS = permute(datax(:, :, us_idx), [2, 1, 3]);%选择超声帧数
            
            Aline = mean(data,3);
            Aline = squeeze(mean(Aline,1));
            [~,DL1] = max(Aline(1:100));
          
            
            % % % % 图像重建的尺寸设置
            % x_size = 50;
            % y_size = 50;
            % z_size = 20;
            % resolution_factor = 10; % 分辨率因子
            
            x_size = 2;
            y_size = 2;
            z_size = 2;
            resolution_factor = 100; % 分辨率因子

            center_x = floor(frame/17) * (-5) + 0.3 + 40;  % 中心坐标X
            center_y = mod(frame-1,17) * 5 - 5.6 - 40; % 中心坐标Y
            center_z = -5;  % 中心坐标Z
            
            %系统参数
            if Is_Denoising == 1 
                data = denoise_sinogram(data);%滤除换能器带宽外的噪声
                dataUS = denoise_sinogram(dataUS);
            end
            
            predelay = -DL1; % 预延迟设置
            pa_data = -data;% 取负是为了让重建背景反色

            % 计算像素点的具体位置
            Npixel_x = x_size * resolution_factor+1;
            Npixel_y = y_size * resolution_factor+1;
            Npixel_z = z_size * resolution_factor+1;
            x_range = ((1:Npixel_x)-(Npixel_x+1)/2)*x_size/(Npixel_x-1) + center_x;
            y_range = ((1:Npixel_y)-(Npixel_y+1)/2)*y_size/(Npixel_y-1) + center_y;
            z_range = ((1:Npixel_z)-(Npixel_z+1)/2)*z_size/(Npixel_z-1) + center_z;
            
            [num_spl, num_rcv, num_rot] = size(dataUS); 
            iq_ch = reshape(hilbert(reshape(dataUS, num_spl, [])), [num_spl, num_rcv, num_rot]);

            fs = 40e6;
            fc = 3.5e6;
            
            c = V_US;
            ang_hole = 16.5/180*pi;
            f_xdc = 15e-3;
            rad_xdc = 108.85e-3;
            
            del_tx = f_xdc/c;
            rad_src = rad_xdc - f_xdc;
            wvl = c/fc;
            x_src = - rad_src * sin(ang_hole);
            y_src = 0;
            z_src = rad_src * cos(ang_hole);
            src = [ x_src, y_src, -z_src ];
            
            rcv = detector * 1e-3;
            rcv(:,3) = -rcv(:,3);
            
            % acq/beamforming specs
            no_rcv = 1:1024;
            del_acq = 100e-6;
            t0 = del_acq;
            
            % mechanical specs
            no_pos = 1:1:100;
            rng_rot = -80; % for crossed hair
            a_rot = 0:-0.8:99*-0.811;
            num_pos = numel(no_pos);

            % imaging volume
            dp = 1e-3 / resolution_factor;
            x_sp = [x_range(1), x_range(end)] * 1e-3; % sp: span
            y_sp = [y_range(1), y_range(end)] * 1e-3;
            z_sp = [z_range(1), z_range(end)] * 1e-3;
                
            x_1d = x_range * 1e-3;
            y_1d = y_range * 1e-3;
            z_1d = z_range * 1e-3;

            num_xp = numel( x_1d );
            num_yp = numel( y_1d );
            num_zp = numel( z_1d );
            [ x_2d , y_2d ] = ndgrid( x_1d , y_1d );
            num_pnt_xy = numel(x_2d);
            
            % beamforming
            % --- 修改 beamforming 部分 ---
            beta = 10/180*pi; 
            fd = 0;
            das_params = [t0, fs, fd, c, beta, del_tx];
            
            corr_ang_x = theta_x;   % 绕X轴旋转 (上下倾斜)
            corr_ang_y = theta_y;   % 绕Y轴旋转 (左右倾斜)
            corr_ang_z = theta_z;   % 绕Z轴旋转 (平面内旋转)
           
            Rx = [1 0 0; 0 cosd(corr_ang_x) -sind(corr_ang_x); 0 sind(corr_ang_x) cosd(corr_ang_x)];
            Ry = [cosd(corr_ang_y) 0 -sind(corr_ang_y); 0 1 0; sind(corr_ang_y) 0 cosd(corr_ang_y)];
            Rz = [cosd(corr_ang_z) -sind(corr_ang_z) 0; sind(corr_ang_z) cosd(corr_ang_z) 0; 0 0 1];
            R_corr = Rz * Ry * Rx; % 全局校正矩阵

            
            iq_im_sum = zeros(num_xp, num_yp, num_zp);
            % iq_image_frame = zeros(num_xp, num_yp, num_zp,num_pos);
            
            for ii = 1 : 1
                % if ~ismember(ii,static_frames)
                %     continue
                % end
                tic;
                % 1. 旋转计算 (保持原样)
                no_pos_i = no_pos(ii);
                a_i = a_rot(no_pos_i);
                R_i = rotz(a_i);
                src_i = (R_i * src')';
                rcv_i = (R_i * rcv(no_rcv, :)')';
            
                src_i = (R_corr * src_i')';
                rcv_i = (R_corr * rcv_i')';
                
                % 2. Update Orientation (保持原样)
                ori_rcv_i = [0, 0, 0] - rcv_i;
                ori_rcv_i = ori_rcv_i ./ vecnorm(ori_rcv_i, 2, 2);
                
                % 3. 调用 CUDA 加速重建 (一行代码替代原来的 jj 循环)
                % 注意：x_1d, y_1d, z_1d 是向量
                iq_im_pos_i = mex_das_gpu(iq_ch(:, no_rcv, no_pos_i), ...
                                          x_1d, y_1d, z_1d, ...
                                          rcv_i, src_i, ori_rcv_i, ...
                                          das_params);
                
                % 4. 累加结果
                % CUDA 输出已经是 (num_xp, num_yp, num_zp) 格式
                % 如果你后面需要保存单帧结果，直接保存 iq_im_pos_i
                iq_im_sum = iq_im_sum + iq_im_pos_i;
                % iq_image_frame(:,:,:,ii) = iq_im_pos_i;
                
                fprintf(1, 'CUDA 3D Recon for Pos %d finished. %.4f sec used.\n', ii, toc);
            end

            dr = Dynamic_Range;
            bm_im = abs(iq_im_sum);
            bm_im = bm_im / max(bm_im, [], 'all');
            bm = 20*log10(bm_im);
            bm(bm<-dr) = -dr;
            bm = bm +dr;
    
            figure(6);
            subplot(131); imagesc(z_sp*1e3, x_sp*1e3, squeeze(max(bm(:,:,:),[],2))); 
            axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
            ylabel('X'); xlabel('Z'); title('XZ proj'); 
            subplot(133); imagesc(z_sp*1e3, y_sp*1e3, squeeze(max(bm(:,:,:),[], 1))); 
            axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
            ylabel('Y'); xlabel('Z'); title('YZ proj'); 
            subplot(132); imagesc(x_sp*1e3, y_sp*1e3, squeeze(max(bm(:,:,:),[], 3))'); 
            axis equal tight; colormap gray; colorbar; 
            ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out'); 

            filenameZX = sprintf('USresult/US Scan zx frame=%d,sos=%.1f.png',frame, c);
            filenameZY = sprintf('USresult/US Scan zy frame=%d,sos=%.1f.png',frame, c);
            filenameXY = sprintf('USresult/US Scan xy frame=%d,sos=%.1f.png',frame, c);
            imwrite(mat2gray(squeeze(max(bm(:,:,:),[],2))), filenameZX);
            imwrite(mat2gray(squeeze(max(bm(:,:,:),[],1))), filenameZY);
            imwrite(mat2gray(squeeze(max(bm(:,:,:),[],3))'), filenameXY);

            disp(frame)

            % 开始计算 FWHM
            pa_img2 = bm;
            [max_val, max_idx] = max(pa_img2(:));
            [idx_x, idx_y, idx_z] = ind2sub(size(pa_img2), max_idx);
            
            % volumeViewer(pa_img2)
            
            directions = {'X', 'Y', 'Z'};
            half_max = max_val / 2;
            
            for dim = 1:3
                % --- A. 根据维度提取数据 ---
                if dim == 1
                    % X方向: 固定 Y 和 Z
                    axis_data = x_range(:); % 强制转为列向量
                    signal_data = double(squeeze(pa_img2(:, idx_y, idx_z)));
                    title_str = 'X-axis Profile';
                elseif dim == 2
                    % Y方向: 固定 X 和 Z
                    axis_data = y_range(:);
                    signal_data = double(squeeze(pa_img2(idx_x, :, idx_z)))'; 
                    % 注意: squeeze后通常是行向量，转置一下保统一
                    signal_data = signal_data(:); 
                    title_str = 'Y-axis Profile';
                else
                    % Z方向: 固定 X 和 Y
                    axis_data = z_range(:);
                    signal_data = double(squeeze(pa_img2(idx_x, idx_y, :)));
                    signal_data = signal_data(:);
                    title_str = 'Z-axis Profile';
                end
                
                % --- B. 计算 FWHM (由峰值向两侧搜索的稳健逻辑) ---
                % 1. 确定当前维度在 1D 剖面线中的主峰索引 (Center Index)
                if dim == 1
                    center_idx = idx_x;
                elseif dim == 2
                    center_idx = idx_y;
                else
                    center_idx = idx_z;
                end
                
                % 2. 向左搜索左边缘
                idx_L = center_idx;
                while idx_L > 1 && signal_data(idx_L) >= half_max
                    idx_L = idx_L - 1;
                end
                
                % 3. 向右搜索右边缘
                idx_R = center_idx;
                while idx_R < length(signal_data) && signal_data(idx_R) >= half_max
                    idx_R = idx_R + 1;
                end
                
                % 4. 检查是否成功找到边界 (防止信号在边界处仍未下降到半高以下)
                if signal_data(idx_L) >= half_max || signal_data(idx_R) >= half_max
                    fwhm_val = NaN; pos_L = NaN; pos_R = NaN;
                    warning('%s 方向信号未完整下降至半高以下，无法计算准确FWHM', directions{dim});
                else
                    % --- 左边缘插值 ---
                    % 此时 idx_L 是半高线外侧(低于半高)的点，idx_L+1 是半高线内侧(高于半高)的点
                    y_low_L = signal_data(idx_L);     y_high_L = signal_data(idx_L+1);
                    x_low_L = axis_data(idx_L);       x_high_L = axis_data(idx_L+1);
                    % 线性插值
                    pos_L = x_low_L + (half_max - y_low_L) * (x_high_L - x_low_L) / (y_high_L - y_low_L);
                    
                    % --- 右边缘插值 ---
                    % 此时 idx_R 是半高线外侧(低于半高)的点，idx_R-1 是半高线内侧(高于半高)的点
                    y_low_R = signal_data(idx_R);     y_high_R = signal_data(idx_R-1);
                    x_low_R = axis_data(idx_R);       x_high_R = axis_data(idx_R-1);
                    % 线性插值
                    pos_R = x_high_R + (half_max - y_high_R) * (x_low_R - x_high_R) / (y_low_R - y_high_R);
                    
                    % 计算宽度
                    fwhm_val = pos_R - pos_L;
                end
                fwhm_result(frame,dim) = fwhm_val;
            end
            disp(fwhm_result(frame,:))
        end
        % writematrix(fwhm_result, 'USresult/US_FWHM.xlsx');

        case 13 % 视场固定，物体平移复合
        %% 子程序说明：
        % 逻辑：传感器阵列和重构视场（Points_img）保持不动
        % 将每一组重构结果粘贴到大矩阵的不同位置，从而形成物体阵列
        imgsize = size(Points_img,1:3);
        pa_img_total = zeros(imgsize+[step_length_y*resolution_factor*(Nframey_scan-mod(Nframey_scan-1,step_y)-1) ...
                                           step_length_x*resolution_factor*(Nframex_scan-mod(Nframex_scan-1,step_x)-1) 0]);    
        [totalsize_y,totalsize_x,totalsize_z] = size(pa_img_total);%最终平移拼接后的图像像素尺寸，单位像素数
        pa_count_total = zeros(totalsize_y, totalsize_x, totalsize_z, 'single');%计数矩阵，用于计算掩膜权重和帧数的影响
        x_range_total = -totalsize_x/resolution_factor/2:totalsize_x/resolution_factor/2;%物理长度坐标，单位mm
        y_range_total = -totalsize_y/resolution_factor/2:totalsize_y/resolution_factor/2;
        z_range_total = -totalsize_z/resolution_factor/2:totalsize_z/resolution_factor/2 + center_z;

        %% --- 绘图窗口初始化 ---
        f9 = figure(9); 
        subplot(131); 
        h_img9_1 = imagesc(x_range, y_range, zeros(length(y_range), length(x_range))); % 初始化空图，拿到句柄 h_img9_1
        axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');
        ylabel('Y'); xlabel('X'); title('pa img XY proj'); set(gca, 'tickdir', 'out');
        
        subplot(132); 
        h_img9_2 = imagesc(x_range, y_range, zeros(length(y_range), length(x_range))); % 拿到句柄 h_img9_2
        axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');
        ylabel('Y'); xlabel('X'); title('GaussianMask XY proj'); set(gca, 'tickdir', 'out');
        
        subplot(133); 
        h_img9_3 = imagesc(x_range, y_range, zeros(length(y_range), length(x_range))); % 拿到句柄 h_img9_3
        axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');
        ylabel('Y'); xlabel('X'); title('pa img masked XY proj'); set(gca, 'tickdir', 'out');
        
        % --- 在循环外初始化 Figure 10 (同理) ---
        f10 = figure(10);
        subplot(131); h_img10_1 = imagesc(z_range_total, x_range_total, zeros(length(x_range_total), length(z_range_total))); % 注意尺寸
        axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');set(gca, 'tickdir', 'out');
        ylabel('X'); xlabel('Z');title('ZX proj'); 
        subplot(132); h_img10_2 = imagesc(x_range_total, y_range_total, zeros(length(y_range_total), length(x_range_total))); % 注意尺寸
        axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');set(gca, 'tickdir', 'out');
        ylabel('Y'); xlabel('X');title('XY proj'); 
        subplot(133); h_img10_3 = imagesc(z_range_total, y_range_total, zeros(length(y_range_total), length(z_range_total))); % 注意尺寸
        axis equal tight; colormap gray; colorbar; axis equal;set(gca, 'YDir', 'normal');set(gca, 'tickdir', 'out');
        ylabel('Y'); xlabel('Z');title('ZY proj'); 

        VM_out_start = VM_out;
        
        for xframe = 1:step_x:Nframex_scan

            for yframe = 1:step_y:Nframey_scan %文件数量方向

                % VM_out = VM_out_start-0.09*(yframe-1 + yframe*(xframe-1)); %水温降低的声速补偿，若实验使用冷水则关闭该补偿
                VM_in = VM_out;
                frame_idx = 1+((xframe-1)*Nframey_scan+(yframe-1))*4;
                [datax,DAQ_time_point] = func_3D_PACT_Data_Time_Read(folder_path,str_name(frame_idx).name);
                
                % 根据表面信号判断区分超声帧和光声帧
                frame1_val = max(sum(datax(:, 1:100, 1)));
                frame2_val = max(sum(datax(:, 1:100, 2)));
                offset = (frame1_val < frame2_val); 
                pa_idx = (1 + offset) : 2 : size(datax, 3); % 光声帧索引
                us_idx = (2 - offset) : 2 : size(datax, 3); % 超声帧索引
                
                pa_data = -datax(:,:,pa_idx);%选择光声帧数
                figure(1);imagesc(pa_data(:,:,1),[-100,100]);

                if Is_Denoising == 1 
                    pa_data = denoise_sinogram(pa_data);%滤除换能器带宽外的噪声
                end

                x_sensor_new = x_sensor;% + (xframe-1)*step_length_x;
                y_sensor_new = y_sensor;% - (yframe-1)*step_length_y;
                pa_data_frame = gpuArray(single(pa_data(:,:,yframe))); % [Nelemt x Nsample]
                detector_new = gpuArray(single([x_sensor_new,y_sensor_new,z_sensor,z_sensor*0+1]));% [Nelemt x 3]
                
                %%
                center_x = -1.75 - 40 + (xframe-1)*step_x*step_length_x;  % 将粒子所在位置设为视场中心坐标X
                center_y = 3.35 - 30 + (yframe-1)*step_y*step_length_y; % 将粒子所在位置设为视场中心坐标Y
                center_z = 5;  % 中心坐标Z
                
                x_range = ((1:Npixel_x)-(Npixel_x+1)/2)*x_size/(Npixel_x-1) + center_x;
                y_range = ((1:Npixel_y)-(Npixel_y+1)/2)*y_size/(Npixel_y-1) + center_y;
                z_range = ((1:Npixel_z)-(Npixel_z+1)/2)*z_size/(Npixel_z-1) + center_z;
                [X_img, Y_img, Z_img] = meshgrid(x_range, y_range, z_range);
                
                X_img = gpuArray(single(X_img));
                Y_img = gpuArray(single(Y_img));
                Z_img = gpuArray(single(Z_img));
                Points_img = cat(4,X_img,Y_img,Z_img);
                
                %%

                pa_total = zeros(size(Points_img(:,:,:,1)),'single');
        
                corr_mat = zeros(Nframe,Nframe);
                % 产生静态帧-门控
                common = mean(pa_data,3);
                tic
        
                for frx = 1:Nframe
                    for fry = frx:Nframe
                        corr_res = corrcoef(pa_data(:,2501:3000,frx),pa_data(:,2501:3000,fry));%选取光声/超声信号范围
                        corr_mat(frx,fry) = corr_res(1,2);%对称相似度矩阵
                    end
                end
                corr_mat = corr_mat+corr_mat';%补全简化计算的部分
                corr_line = mean(corr_mat,1);
                corr_line = corr_line/max(corr_line(3:end));%避免静止帧相关性太强导致归一化后系数偏小
                corr_line(1) = 10; %先采集后旋转时，强制首帧参考
                corr_line(1:9) = 0; %先采集后旋转时，去除未旋转的数据
                static_frames = 1:Nframe;
                top_vals = maxk(corr_line, 20);%找出最大的20个值，并按降序排列
                Similarity_threshold = top_vals(end);%动态调整复合时所用的相似度阈值，避免不同组数据相似度波动导致复合帧数不同
                static_frames = static_frames(corr_line>=Similarity_threshold);
                % 绘制相关系数图
                figure(11),plot(corr_line,'b'),hold on
                for isf = static_frames
                    plot(isf,corr_line(isf),'*r'),hold on
                end
                hold off;
                
                
                % 软件中触发速度11000对应0.800°，那么软件中触发速度10000可线性计算-改
                delta_angle = -11000*0.800/11000; %角度°
                static_Nframe = size(static_frames,2);

                firstframe_flag = 1; %首帧标识 1：是首帧，0：非首帧

                for frame = 1:1:static_Nframe %注意这里的step*delta_angle为相邻复合角
            
                    tic
                     % 坐标旋转(角度制,顺时针为负，逆时针为正 ：对应Y轴正方向朝上，否则相反）
                    theta_x = 0; % 以x轴为中心旋转 
                    theta_y = 0; % 以y轴为中心旋转 
                    theta_z = (static_frames(frame)-static_frames(1))*delta_angle; % 以z轴为中心旋转 
                        
                    %坐标整体平移(这里的坐标平移会影响椭球中心选取)
                    trans_x = 0;
                    trans_y = 0;
                    trans_z = 0;
                    
                    rotate_x_mat = [1 0 0 0;0 cosd(theta_x) -sind(theta_x),0;0 sind(theta_x) cosd(theta_x) 0;0 0 0 1];
                    rotate_y_mat = [cosd(theta_y) 0 -sind(theta_y) 0;0 1 0,0;sind(theta_y) 0 cosd(theta_y) 0;0 0 0 1];
                    rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0;sind(theta_z) cosd(theta_z) 0,0;0 0 1 0;0 0 0 1];
                    trans_mat = [1 0 0 trans_x;0 1 0 trans_y;0 0 1 trans_z;0 0 0 1];
                    
                    afine_mat = trans_mat*rotate_x_mat*rotate_y_mat*rotate_z_mat; %以原点为中心，先转z轴，再转y轴，再转x轴，最后平移
                    detector_corr=detector_new*afine_mat'; 
        
                    pa_data_frame = gpuArray(single(pa_data(:,:,static_frames(frame)))); % [Nelemt x Nsample]
                    Points_sensor_all = gpuArray(single(detector_corr(:,1:3)));% [Nelemt x 3]
                    
                    % 调用 CUDA MEX 函数
                    [pa_img, total_angle_weight] = DualSpeedReconstraction_mex([Ellipse.a,Ellipse.b,Ellipse.c,Ellipse.centerx,Ellipse.centery,Ellipse.centerz], ...
                                                                        Points_sensor_all, Points_img, pa_data_frame, ...
                                                                        single(fs), single(predelay), single(VM_out), single(VM_in), single(R));
                     
                    disp(['xframe : ',num2str(xframe),' yframe : ',num2str(yframe),' frame : ',num2str(frame)]);
                    
                    % 收集GPU数据
                    pa_img1 = gather(pa_img);
                    total_angle_weight = gather(total_angle_weight);
                    
                    % 归一化并应用非线性增强
                    pa_img2 = pa_img1./total_angle_weight;
                    [GaussianMask, ~, ~] = generateGaussianMask({x_range, y_range}, 'Center', [center_x,center_y], 'Sigma', 9);
                    GaussianMask = GaussianMask.*ones(1,1,size(z_range,2));%z方向均匀
                    pa_img3 = pa_img2.*GaussianMask;
                    % pa_img3 = pa_img3/max(pa_img3,[],'all');
                    % pa_img3 = pa_img2.*(2-GaussianMask); %取消高斯滤波并进行高斯增强
        
                    if firstframe_flag == 0 %判断是否为首帧
                        %图像配准
                        % pa_img3 = rigidRegistration3D(pa_ref, pa_img3);
                    else
                        pa_ref = pa_img3;%固定首帧作为参考帧
                    end

                    pa_total = pa_total+subplus(pa_img3);
                    firstframe_flag = 0;%关闭首帧标识
                    toc
        
                    % 显示不同视角的图像
                    imin=0;
                    imax=max(pa_total,[],"all");
                    % --- 通过句柄更新数据，不会抢焦点 ---
                    set(h_img9_1, 'CData', squeeze(max(pa_img2(:,:,:),[], 3)));
                    set(h_img9_2, 'CData', squeeze(max(GaussianMask(:,:,:),[], 3)));
                    set(h_img9_3, 'CData', squeeze(max(pa_total(:,:,:),[], 3)));
                                     
                    drawnow limitrate;

                    filenameZX = sprintf('zx xframe=%d, yframe=%d, frame=%d.png', xframe, yframe, frame);
                    filenameZY = sprintf('zy xframe=%d, yframe=%d, frame=%d.png', xframe, yframe, frame);
                    filenameXY = sprintf('xy xframe=%d, yframe=%d, frame=%d.png', xframe, yframe, frame);
                    imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],1))), filenameZX);
                    imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],2))), filenameZY);
                    imwrite(mat2gray(squeeze(max(pa_total(end:-1:1,:,:),[],3))), filenameXY);

                end % frame end

                toc
                SubWinSize_y = imgsize(1)-1;
                SubWinSize_x = imgsize(2)-1;
                SubWinSize_z = imgsize(3)-1;

                idx_y = totalsize_y-(yframe-1)*step_length_y*resolution_factor-SubWinSize_y...
                       :totalsize_y-(yframe-1)*step_length_y*resolution_factor;
                idx_x = (xframe-1)*step_length_x*resolution_factor+1 ...
                       :(xframe-1)*step_length_x*resolution_factor+1+SubWinSize_x;

                current_weight = GaussianMask * single(static_Nframe);%单组数据权重

                pa_count_total(idx_y,idx_x,:) = pa_count_total(idx_y,idx_x,:) + current_weight;%计算每帧图像累加贡献的权重矩阵

                pa_img_total(idx_y,idx_x,:) = pa_img_total(idx_y,idx_x,:) + pa_total;
                pa_img_total_2 = pa_img_total ./ (pa_count_total + eps);%去除帧数不均对亮度的影响，避免除以0
                imin=min(pa_img_total_2,[],"all");
                imax=max(pa_img_total_2,[],"all");

                % 更新 Figure 10 的数据
                set(h_img10_1, 'CData', squeeze(max(pa_img_total_2(:,:,:),[],1)));
                set(h_img10_1.Parent, 'CLim', [imin, imax]); 
                set(h_img10_2, 'CData', squeeze(max(pa_img_total_2(:,:,:),[],3)));
                set(h_img10_2.Parent, 'CLim', [imin, imax]); 
                set(h_img10_3, 'CData', squeeze(max(pa_img_total_2(:,:,:),[],2)));
                set(h_img10_3.Parent, 'CLim', [imin, imax]); 
                % drawEllipsoidOverlay(Ellipse);%显示双声速范围
                
                drawnow limitrate;
                            
                filenameZX = sprintf('step=%d zx xframe=%d, yframe=%d.png',step_x, xframe, yframe);
                filenameZY = sprintf('step=%d zy xframe=%d, yframe=%d.png',step_x, xframe, yframe);
                filenameXY = sprintf('step=%d xy xframe=%d, yframe=%d.png',step_x, xframe, yframe);
                imwrite(mat2gray(squeeze(max(pa_img_total_2(end:-1:1,:,:),[],1))), filenameZX);
                imwrite(mat2gray(squeeze(max(pa_img_total_2(end:-1:1,:,:),[],2))), filenameZY);
                imwrite(mat2gray(squeeze(max(pa_img_total_2(end:-1:1,:,:),[],3))), filenameXY);
            end %yframe end
        end %xframe end
       save(['pa_img_fullsize_step',num2str(step_x),'.mat'], 'pa_img_total_2', '-v7.3');
    otherwise

        disp('Error: Undefined reconstruct mode!');

end

%% 可视化
% pa_total = sum(pa_img_frames(:,:,:,1:20),4);
% pa_total = pa_img2;
% 
% figure(3); 
% set (gca,'position',[0.1,0.1,0.8,0.8]);
% % subplot(131); imagesc(z_range, x_range, squeeze(max(pa_total_flip(:,:,:),[],2)),[imin,imax]); 
% subplot(131); imagesc(squeeze(max(pa_total(:,:,:),[],2))); 
% axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
% ylabel('X'); xlabel('Z'); title('XZ proj'); 
% % subplot(132); imagesc(x_range, y_range, squeeze(max(pa_total_flip(:,:,:),[], 3)),[imin,imax]); 
% subplot(132); imagesc(squeeze(max(pa_total(:,:,:),[], 3))); 
% axis equal tight; colormap gray; colorbar; 
% ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out'); 
% % subplot(133); imagesc(z_range, y_range, squeeze(max(pa_total_flip(:,:,:),[], 1)),[imin,imax]); 
% subplot(133); imagesc(squeeze(max(pa_total(:,:,:),[], 1))); 
% axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
% ylabel('Y'); xlabel('Z'); title('YZ proj'); 
% 
% 
% dr = 20;
% iq_total = sum(iq_image_frame,4);
% 
% bm_im = abs(iq_total);
% bm_im = bm_im / max(bm_im, [], 'all');
% bm = 20*log10(bm_im);
% bm(bm<-dr) = -dr;
% bm_total = bm +dr;
% 
% figure(2); 
% % set(gca,'position',[0.1,0.1,0.8,0.8]);
% subplot(131); imagesc(squeeze(max(bm_total(:,:,:),[],1))); 
% axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
% ylabel('X'); xlabel('Z'); title('XZ proj'); 
% subplot(132); imagesc(squeeze(max(bm_total(:,:,:),[], 3))'); 
% axis equal tight; colormap gray; colorbar; 
% ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out'); 
% subplot(133); imagesc(squeeze(max(bm_total(:,:,:),[], 2))); 
% axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
% ylabel('Y'); xlabel('Z'); title('YZ proj'); 


