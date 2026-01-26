% ************************* 3D reconstruction code ************************
%
% 本程序用于球阵光声探测器阵列数据3D重建
% 可选单声速重建或双声速重建
% 
% 椭球面位置需根据实验情况进行调整
%
% Author：Xiali Gao  Version：3.0  2025.12.10
%
% *************************************************************************

%% 初始化及加载文件
clc; 
% clearvars -except data; 
% clear;
% close all; 
gpuDevice(1).reset() % 重置GPU，释放所有显存

addpath 'D:\GXL\DataRead';
addpath 'E:\Summer Pear\Program\work\matlab\Photot Acoustic Imaging\3DPACT_RECON';
% folder_path = strcat('D:\GXL\hand\SmallUnet\');

folder_path = 'E:\Summer Pear\Program\work\matlab\Photot Acoustic Imaging\球阵系统\25.12.24\旋转\正常高度\';
str_name = dir(fullfile(folder_path, '*.mat'));
[datax,DAQ_time_point] = func_3D_PACT_Data_Time_Read(folder_path,str_name(1).name);

data = datax(:,:,1:2:end);%选择光声帧数
Aline = mean(data,3);
Aline = squeeze(mean(Aline,1));
[~,DL1] = max(Aline(1:100));

% load finger3.mat % 加载数据文件
% data = finger3(:,:,:);
% Aline = mean(data,3);
% Aline = squeeze(mean(Aline,1));
% [~,DL1] = max(Aline(1:100));
% % DL1 = 42;

dector=load('coordinate.txt'); % 加载探测器坐标文件
% dector = dector(1:3:end,:,:);
% data = data(1:3:end,:,:);
[Nelemt, Nsample, Nframe] = size(data); % 获取样本数和元素数

figure(1);imagesc(-data(:,:,1));colormap gray;

%% 系统参数设置
reconstruct_mode = 6; % 1: 单声速CUDA重建; 2: 双声速CUDA重建; 3：内声速迭代 
                      % 4：外声速迭代; 5:单声速迭代 6:单声速旋转覆合 7:双声速旋转覆合 
                      % 8:单声速相干因子旋转覆合 9:单声速二维平移覆合 10：双声速旋转平移覆合

% 声速设置
T = 22.8; % 水温
V_M = 1522.0;
V_M_Range = 1529:0.5:1560; % 单声速迭代范围

% VM_out = waterSoundSpeed(T); % 外声速（水），单位 m/s 
VM_out = 1480; % 外声速（水），单位 m/s  
VM_out_Range = 1475:0.5:1499; % 外声速迭代范围
VM_in = 1610; % 内声速，单位 m/s  
VM_in_Range = 1555:5:1699; % 内声速迭代范围


step_x = 1; %mm 平移覆合位移数（文件数量方向）
step_y = 1; %mm 平移覆合位移数(帧方向）
step_length_x = 10; %mm 水平相邻两帧位移距离
step_length_y = 10; %mm 垂直相邻两帧位移距离

% 图像重建的尺寸设置
x_size = 40-10;
y_size = 80-40;
z_size = 40-10;
resolution_factor = 10; % 分辨率因子
center_x = 20-20;  % 中心坐标X
center_y = -30+30; % 中心坐标Y
center_z = 10-10;  % 中心坐标Z

%椭球面参数设置
Ellipse.a = 15.0; % 椭球面x轴半径
Ellipse.b = 28.0;  % 椭球面y轴半径
Ellipse.c = 13.5; % 椭球面z轴半径
Ellipse.centerx = -2.8; % 椭球面中心坐标
Ellipse.centery = -2.0;
Ellipse.centerz = 7.2;

%系统参数
predelay = -DL1; % 预延迟设置
pa_data = -data;% 取负是为了让重建背景反色
fs = 40; % 声音采样频率 单位为MHz
R = 100; % 球形探测器阵列半径

% %对球阵传感器阵列插值
% [pa_data, dector] = interpolation(pa_data(:,:,1),dector);

% 计算像素点的具体位置
Npixel_x = x_size * resolution_factor+1;
Npixel_y = y_size * resolution_factor+1;
Npixel_z = z_size * resolution_factor+1;
x_range = ((1:Npixel_x)-(Npixel_x+1)/2)*x_size/(Npixel_x-1) + center_x;
y_range = ((1:Npixel_y)-(Npixel_y+1)/2)*y_size/(Npixel_y-1) + center_y;
z_range = ((1:Npixel_z)-(Npixel_z+1)/2)*z_size/(Npixel_z-1) + center_z;

% 创建三维网格坐标
[X_img, Y_img, Z_img] = meshgrid(x_range, y_range, z_range);

% 坐标旋转(角度制,顺时针为正，逆时针为负）
theta_x = 0; % 以x轴为中心旋转 
theta_y = 0; % 以y轴为中心旋转 
theta_z = 42.6; % 以z轴为中心旋转 位移台与坐标轴夹角38.87

%坐标平移(这里的坐标平移会影响椭球中心选取)
trans_x = 0;
trans_y = 0;
trans_z = 0;

rotate_x_mat = [1 0 0 0;0 cosd(theta_x) -sind(theta_x),0;0 sind(theta_x) cosd(theta_x) 0;0 0 0 1];
rotate_y_mat = [cosd(theta_y) 0 -sind(theta_y) 0;0 1 0,0;sind(theta_y) 0 cosd(theta_y) 0;0 0 0 1];
rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0;sind(theta_z) cosd(theta_z) 0,0;0 0 1 0;0 0 0 1];
trans_mat = [1 0 0 trans_x;0 1 0 trans_y;0 0 1 trans_z;0 0 0 1];
afine_mat = trans_mat*rotate_x_mat*rotate_y_mat*rotate_z_mat; %以原点为中心，先转z轴，再转y轴，再转x轴，最后平移

dector_new=[dector,dector(:,1)*0+1]*afine_mat'; 

% 获取并调整传感器位置，Z 坐标取负以调整坐标系统
x_sensor=dector_new(:,1);
y_sensor=dector_new(:,2);
z_sensor=-dector_new(:,3); % 注意此处z轴坐标反转

% 将传感器坐标移至GPU以加速计算
x_sensor = gpuArray(single(x_sensor));
y_sensor = gpuArray(single(y_sensor));
z_sensor = gpuArray(single(z_sensor));

% 将图像坐标移至GPU
X_img = gpuArray(single(X_img));
Y_img = gpuArray(single(Y_img));
Z_img = gpuArray(single(Z_img));
Points_img = cat(4,X_img,Y_img,Z_img);

%% --- 绘图窗口初始化 ---
f9 = figure(9); 
subplot(131); 
h_img9_1 = imagesc(x_range, y_range, zeros(length(y_range), length(x_range))); % 初始化空图，拿到句柄 h_img9_1
axis equal tight; colormap gray; colorbar; 
ylabel('Y'); xlabel('X'); title('pa img XY proj'); set(gca, 'tickdir', 'out');

subplot(132); 
h_img9_2 = imagesc(x_range, y_range, zeros(length(y_range), length(x_range))); % 拿到句柄 h_img9_2
axis equal tight; colormap gray; colorbar; 
ylabel('Y'); xlabel('X'); title('GaussianMask XY proj'); set(gca, 'tickdir', 'out');

subplot(133); 
h_img9_3 = imagesc(x_range, y_range, zeros(length(y_range), length(x_range))); % 拿到句柄 h_img9_3
axis equal tight; colormap gray; colorbar; 
ylabel('Y'); xlabel('X'); title('pa img masked XY proj'); set(gca, 'tickdir', 'out');

% --- 在循环外初始化 Figure 10 (同理) ---
f10 = figure(10);
subplot(131); h_img10_1 = imagesc(z_range, x_range, zeros(length(x_range), length(z_range))); % 注意尺寸
axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out');
ylabel('X'); xlabel('Z');title('ZX proj'); 
subplot(132); h_img10_2 = imagesc(x_range, y_range, zeros(length(y_range), length(x_range))); % 注意尺寸
axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out');
ylabel('Y'); xlabel('X');title('XY proj'); 
subplot(133); h_img10_3 = imagesc(z_range, y_range, zeros(length(y_range), length(z_range))); % 注意尺寸
axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out');
ylabel('Y'); xlabel('Z');title('ZY proj'); 

%% 回波信号数据处理的主循环
tic % 开始计时
switch reconstruct_mode

    case 1 %单声速重建
        for frame = 1%:Nframe %设置双声速重建区域时，最好使用双声速重建复合的第一帧确定参数
            tic
            pa_data_frame = gpuArray(single(pa_data(:,:,frame))); % [Nelemt x Nsample]
            Points_sensor_all = gpuArray(single([x_sensor,y_sensor,z_sensor]));% [Nelemt x 3]
             
            % %对球阵传感器阵列插值
            % [pa_data_frame, Points_sensor_all] = interpolation(pa_data_frame,Points_sensor_all);

            % 调用 CUDA MEX 函数
            [pa_img, total_angle_weight] = SingleSpeedReconstraction_mex(Points_sensor_all, Points_img, pa_data_frame, single(fs), single(predelay), single(V_M), single(R));            % % 初始化图像缓冲区和角度权重

            disp(['frame : ',num2str(frame)]);
            toc
            % 收集GPU数据
            pa_img1 = gather(pa_img);
            total_angle_weight = gather(total_angle_weight);
            
            % 归一化并应用非线性增强
            pa_img2 = subplus(pa_img1)./total_angle_weight;
            
            % 3D图像查看
            % volumeViewer(pa_img2);
            
            % 找到最大强度点的位置
            % [ym, xm, zm] = ind2sub(size(pa_img2), find(pa_img2 == max(pa_img2(:)))); 
            
            % 显示不同视角的图像
            imin=min(pa_img2,[],"all");
            imax=max(pa_img2,[],"all");
            
            figure(2); set (gca,'position',[0.1,0.1,0.8,0.8]);
            subplot(131); imagesc(z_range, x_range, squeeze(max(pa_img2(:,:,:),[],1)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
            ylabel('X'); xlabel('Z'); title('XZ proj'); 
            subplot(133); imagesc(z_range, y_range, squeeze(max(pa_img2(:,:,:),[], 2)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
            ylabel('Y'); xlabel('Z'); title('YZ proj'); 
            subplot(132); imagesc(x_range, y_range, squeeze(max(pa_img2(:,:,:),[], 3)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; 
            ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out'); 
            drawEllipsoidOverlay(Ellipse);%显示双声速范围

            filenameZX = sprintf('zx frame=%d,V_M=%.1f.png',frame, V_M);
            filenameZY = sprintf('zy frame=%d,V_M=%.1f.png',frame, V_M);
            filenameXY = sprintf('xy frame=%d,V_M=%.1f.png',frame, V_M);
            imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],1))), filenameZX);
            imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],2))), filenameZY);
            imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],3))), filenameXY);

            

        end

    case 2 %双声速cuda重建
        for frame = 1%:Nframe
            tic
            pa_data_frame = gpuArray(single(pa_data(:,:,frame))); % [Nelemt, Nsample]
            Points_sensor_all = gpuArray(single([x_sensor,y_sensor,z_sensor]));% [Nelemt x 1]合并为[Nelemt x 3]

            % pa_data_frame = resize(pa_data_frame,[1024*2 4384]);
            % Points_sensor_all = resize(Points_sensor_all,[1024*2,3]);
    
            % 调用新的MEX函数一次性计算所有传感器对图像的贡献
            [pa_img, total_angle_weight] = DualSpeedReconstraction_mex([Ellipse.a,Ellipse.b,Ellipse.c,Ellipse.centerx,Ellipse.centery,Ellipse.centerz], ...
                                                                Points_sensor_all, Points_img, pa_data_frame, ...
                                                                single(fs), single(predelay), single(VM_out), single(VM_in), single(R));
            % [pa_img, total_angle_weight] = DualSpeedReconstraction_focus_mex([Ellipse.a,Ellipse.b,Ellipse.c,Ellipse.centerx,Ellipse.centery,Ellipse.centerz], ...
            %                                                     Points_sensor_all, Points_img, pa_data_frame, ...
            %                                                     single(fs), single(predelay), single(VM_out), single(VM_in), single(R),single(0.01));

            pa_img1 = gather(pa_img);
            total_angle_weight = gather(total_angle_weight);
    
            pa_img2 = subplus(pa_img1)./total_angle_weight;
    
            disp(['frame : ',num2str(frame)]);
            toc
            % 显示不同视角的图像
            imin=min(pa_img2,[],"all");
            imax=max(pa_img2,[],"all");
            
            figure(1);         
            subplot(131); imagesc(z_range, x_range, squeeze(max(pa_img2(:,:,:),[],1)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
            ylabel('X'); xlabel('Z'); title('XZ proj'); 
            subplot(133); imagesc(z_range, y_range, squeeze(max(pa_img2(:,:,:),[], 2)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
            ylabel('Y'); xlabel('Z'); title('YZ proj'); 
            subplot(132); imagesc(x_range, y_range, squeeze(max(pa_img2(:,:,:),[], 3)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; 
            ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out'); 

            filenameZX = sprintf('zx frame=%d,VM_out=%.1f,VM_in=%.1f.png',frame, VM_out, VM_in);
            filenameZY = sprintf('zy frame=%d,VM_out=%.1f,VM_in=%.1f.png',frame, VM_out, VM_in);
            filenameXY = sprintf('xy frame=%d,VM_out=%.1f,VM_in=%.1f.png',frame, VM_out, VM_in);
            imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],1))), filenameZX);
            imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],2))), filenameZY);
            imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],3))), filenameXY);
        end

    case 3 %内声速循环
        for VM_in = VM_in_Range 
            for frame = 1
                pa_data_frame = gpuArray(single(pa_data(:,:,frame))); % [Nelemt, Nsample]
                Points_sensor_all = gpuArray(single([x_sensor,y_sensor,z_sensor]));% [Nelemt x 1]合并为[Nelemt x 3]
                
                tic
                % 调用新的MEX函数一次性计算所有传感器对图像的贡献
                [pa_img, total_angle_weight] = DualSpeedReconstraction_mex([Ellipse.a,Ellipse.b,Ellipse.c,Ellipse.centerx,Ellipse.centery,Ellipse.centerz], ...
                                                                Points_sensor_all, Points_img, pa_data_frame, ...
                                                                single(fs), single(predelay), single(VM_out), single(VM_in), single(R));
                toc

                % 收集GPU数据
                pa_img1 = gather(pa_img);
                total_angle_weight = gather(total_angle_weight);
        
                 % 归一化并应用非线性增强
                pa_img2 = subplus(pa_img1)./total_angle_weight;

                imin=1;
                imax = max(pa_img2,[],"all");
                k = 1;

                figure(3);        
                subplot(131); imagesc(z_range, x_range, squeeze(max(pa_img2(:,:,:),[],1)),[imin,imax*k]); 
                axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
                ylabel('X'); xlabel('Z'); title('XZ proj'); 
                subplot(133); imagesc(z_range, y_range, squeeze(max(pa_img2(:,:,:),[], 2)),[imin,imax*k]); 
                axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
                ylabel('Y'); xlabel('Z'); title('YZ proj'); title(['VM_in = ',num2str(VM_in)]); 
                subplot(132); imagesc(x_range, y_range, squeeze(max(pa_img2(:,:,:),[], 3)),[imin,imax*k]); 
                axis equal tight; colormap gray; colorbar; 
                ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out'); 
                set(gca, 'tickdir', 'out');title(['VM_out = ',num2str(VM_out)]); 
                
                filenameZX = sprintf('zx VM_out=%.1f,VM_in=%.1f.png', VM_out,VM_in);
                filenameZY = sprintf('zy VM_out=%.1f,VM_in=%.1f.png', VM_out,VM_in);
                filenameXY = sprintf('xy VM_out=%.1f,VM_in=%.1f.png', VM_out,VM_in);
                imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],1))), filenameZX);
                imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],2))), filenameZY);
                imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],3))), filenameXY);

            end % 结束帧循环
        end % 结束声速循环

    case 4 %外声速循环
        for VM_out = VM_out_Range 
            for frame = 1
                pa_data_frame = gpuArray(single(pa_data(:,:,frame))); % [Nelemt, Nsample]
                Points_sensor_all = gpuArray(single([x_sensor,y_sensor,z_sensor]));% [Nelemt x 1]合并为[Nelemt x 3]
                
                tic
                % 调用新的MEX函数一次性计算所有传感器对图像的贡献
                [pa_img, total_angle_weight] = DualSpeedReconstraction_mex([Ellipse.a,Ellipse.b,Ellipse.c,Ellipse.centerx,Ellipse.centery,Ellipse.centerz], ...
                                                                Points_sensor_all, Points_img, pa_data_frame, ...
                                                                single(fs), single(predelay), single(VM_out), single(VM_in), single(R));
                toc

                % 收集GPU数据
                pa_img1 = gather(pa_img);
                total_angle_weight = gather(total_angle_weight);
        
                % 归一化并应用非线性增强
                pa_img2 = subplus(pa_img1)./total_angle_weight;

                imin=1;
                imax = max(pa_img2,[],"all");
                k = 1;

                figure(3);        
                subplot(131); imagesc(z_range, x_range, squeeze(max(pa_img2(:,:,:),[],1)),[imin,imax*k]); 
                axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
                ylabel('X'); xlabel('Z'); title('XZ proj'); 
                subplot(133); imagesc(z_range, y_range, squeeze(max(pa_img2(:,:,:),[], 2)),[imin,imax*k]); 
                axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
                ylabel('Y'); xlabel('Z'); title('YZ proj'); title(['VM_in = ',num2str(VM_in)]); 
                subplot(132); imagesc(x_range, y_range, squeeze(max(pa_img2(:,:,:),[], 3)),[imin,imax*k]); 
                axis equal tight; colormap gray; colorbar; 
                ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out'); 
                set(gca, 'tickdir', 'out');title(['VM_out = ',num2str(VM_out)]); 
                
                filenameZX = sprintf('out zx VM_out=%.1f,VM_in=%.1f.png', VM_out,VM_in);
                filenameZY = sprintf('out zy VM_out=%.1f,VM_in=%.1f.png', VM_out,VM_in);
                filenameXY = sprintf('out xy VM_out=%.1f,VM_in=%.1f.png', VM_out,VM_in);
                imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],1))), filenameZX);
                imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],2))), filenameZY);
                imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],3))), filenameXY);

            end % 结束帧循环
        end % 结束声速循环

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
                axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
                ylabel('X'); xlabel('Z'); title('XZ proj'); 
                subplot(133); imagesc(z_range, y_range, squeeze(max(pa_img2(:,:,:),[], 2)),[imin,imax*k]); 
                axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
                ylabel('Y'); xlabel('Z'); title('YZ proj'); 
                subplot(132); imagesc(x_range, y_range, squeeze(max(pa_img2(:,:,:),[], 3)),[imin,imax*k]); 
                axis equal tight; colormap gray; colorbar; 
                ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out'); 
                set(gca, 'tickdir', 'out');title(['VM = ',num2str(V_M)]); 
                
                filenameZX = sprintf('zx VM=%.1f.png', V_M);
                filenameZY = sprintf('zy VM=%.1f.png', V_M);
                filenameXY = sprintf('xy VM=%.1f.png', V_M);
                imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],1))), filenameZX);
                imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],2))), filenameZY);
                imwrite(mat2gray(squeeze(max(pa_img2(:,:,:),[],3))), filenameXY);
    
            end
        end

    case 6 %单声速多帧旋转覆合
        pa_total = zeros(size(Points_img(:,:,:,1)),'single');

        % 软件中触发速度11000对应0.811°，那么软件中触发速度5000可线性计算
        delta_angle = -11000*0.811/11000; %角度°

        % 坐标旋转(角度制,顺时针为正，逆时针为负）
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

        firstframe_flag = 1; %首帧标识 1：是首帧，0：非首帧

        for frame = 1:1:Nframe
            
            dector_new=dector_new*afine_mat'; 

            % 获取并调整传感器位置，Z 坐标取负以调整坐标系统
            x_sensor=dector_new(:,1);
            y_sensor=dector_new(:,2);
            z_sensor=-dector_new(:,3); % 注意此处z轴坐标反转
            
            % 将传感器坐标移至GPU以加速计算
            x_sensor = gpuArray(single(x_sensor));
            y_sensor = gpuArray(single(y_sensor));
            z_sensor = gpuArray(single(z_sensor));

            pa_data_frame = gpuArray(single(pa_data(:,:,frame))); % [Nelemt x Nsample]
            Points_sensor_all = gpuArray(single([x_sensor,y_sensor,z_sensor]));% [Nelemt x 3]
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
            axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
            ylabel('X'); xlabel('Z'); title('XZ proj'); 
            subplot(133); imagesc(z_range, y_range, squeeze(max(pa_total(:,:,:),[],2))); 
            axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
            ylabel('Y'); xlabel('Z'); title('YZ proj'); 
            subplot(132); imagesc(x_range, y_range, squeeze(max(pa_total(:,:,:),[],3))); 
            axis equal tight; colormap gray; colorbar; 
            ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out'); 

            filenameZX = sprintf('zx frame=%d,V_M=%.1f.png',frame, V_M);
            filenameZY = sprintf('zy frame=%d,V_M=%.1f.png',frame, V_M);
            filenameXY = sprintf('xy frame=%d,V_M=%.1f.png',frame, V_M);
            imwrite(mat2gray(squeeze(max(pa_total(:,:,:),[],1))), filenameZX);
            imwrite(mat2gray(squeeze(max(pa_total(:,:,:),[],2))), filenameZY);
            imwrite(mat2gray(squeeze(max(pa_total(:,:,:),[],3))), filenameXY);

        end

    case 7 %双声速覆合
        pa_total = zeros(size(Points_img(:,:,:,1)));
        % FOV = [x_size,y_size,z_size];
        % center = [center_x,center_y,center_z];
        corr_mat = zeros(Nframe,Nframe);
        % 产生静态帧-门控-改
        common = mean(pa_data,3);
        tic

        for frx = 1:Nframe
            for fry = frx:Nframe
                corr_res = corrcoef(pa_data(:,2201:3000,frx),pa_data(:,2201:3000,fry));
                corr_mat(frx,fry) = corr_res(1,2);%对称相似度矩阵
            end
        end
        corr_mat = corr_mat+corr_mat';%补全简化计算的部分
        corr_line = mean(corr_mat,1);
        corr_line = corr_line/max(corr_line(:));
        static_frames = 1:Nframe;
        static_frames = static_frames(find(corr_line>=0.85));
        figure,plot(corr_line,'b'),hold on, 
        for isf = static_frames
            plot(isf,corr_line(isf),'*r'),hold on,
        end
        hold off;
        toc
        
        % 软件中触发速度11000对应0.811°，那么软件中触发速度10000可线性计算-改
        delta_angle = -11000*0.811/11000; %角度°
        static_Nframe = size(static_frames,2);

        firstframe_flag = 1; %首帧标识 1：是首帧，0：非首帧

        for frame = 1:1:static_Nframe %注意这里的step*delta_angle为相邻复合角
            
            % 坐标旋转(角度制,顺时针为正，逆时针为负）
            theta_x = 0; % 以x轴为中心旋转 
            theta_y = 0; % 以y轴为中心旋转 
            theta_z = (static_frames(frame)-static_frames(1))*delta_angle; % 以z轴为中心旋转 
            
            %坐标平移(这里的坐标平移会影响椭球中心选取)
            trans_x = 0;
            trans_y = 0;
            trans_z = 0;
            
            rotate_x_mat = [1 0 0 0;0 cosd(theta_x) -sind(theta_x),0;0 sind(theta_x) cosd(theta_x) 0;0 0 0 1];
            rotate_y_mat = [cosd(theta_y) 0 -sind(theta_y) 0;0 1 0,0;sind(theta_y) 0 cosd(theta_y) 0;0 0 0 1];
            rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0;sind(theta_z) cosd(theta_z) 0,0;0 0 1 0;0 0 0 1];
            trans_mat = [1 0 0 trans_x;0 1 0 trans_y;0 0 1 trans_z;0 0 0 1];
            afine_mat = trans_mat*rotate_x_mat*rotate_y_mat*rotate_z_mat; %以原点为中心，先转z轴，再转y轴，再转x轴，最后平移
            
            dector_corr=dector_new*afine_mat'; 
            % 获取并调整传感器位置，Z 坐标取负以调整坐标系统
            x_sensor=dector_corr(:,1);
            y_sensor=dector_corr(:,2);
            z_sensor=-dector_corr(:,3); % 注意此处z轴坐标反转
            
            % 将传感器坐标移至GPU以加速计算
            x_sensor = gpuArray(single(x_sensor));
            y_sensor = gpuArray(single(y_sensor));
            z_sensor = gpuArray(single(z_sensor));

            pa_data_frame = gpuArray(single(pa_data(:,:,static_frames(frame)))); % [Nelemt x Nsample]
            Points_sensor_all = gpuArray(single([x_sensor,y_sensor,z_sensor]));% [Nelemt x 3]
            tic
            % 调用 CUDA MEX 函数
            [pa_img, total_angle_weight] = DualSpeedReconstraction_mex([Ellipse.a,Ellipse.b,Ellipse.c,Ellipse.centerx,Ellipse.centery,Ellipse.centerz], ...
                                                                Points_sensor_all, Points_img, pa_data_frame, ...
                                                                single(fs), single(predelay), single(VM_out), single(VM_in), single(R));
             toc
            disp(['frame : ',num2str(frame)]);
            
            % 收集GPU数据
            pa_img1 = gather(pa_img);
            total_angle_weight = gather(total_angle_weight);
            
            % 归一化并应用非线性增强
            pa_img2 = pa_img1./total_angle_weight;

            if firstframe_flag == 0 %判断是否为首帧
                %图像配准
                pa_img2 = rigidRegistration3D(pa_ref, pa_img2);

            else
                pa_ref = pa_img2;%固定首帧作为参考帧
    
            end
            pa_total = pa_total+pa_img2;
            firstframe_flag = 0;%关闭首帧标识

            % 显示不同视角的图像
            imin=0;
            imax=max(pa_total,[],"all");
            
            figure(7); set (gca,'position',[0.1,0.1,0.8,0.8]);
            subplot(131); imagesc(z_range, x_range, squeeze(max(pa_total(:,:,:),[],1)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
            ylabel('X'); xlabel('Z'); title('XZ proj'); 
            subplot(133); imagesc(z_range, y_range, squeeze(max(pa_total(:,:,:),[], 2)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
            ylabel('Y'); xlabel('Z'); title('YZ proj'); 
            subplot(132); imagesc(x_range, y_range, squeeze(max(pa_total(:,:,:),[], 3)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; 
            ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out'); 
            drawEllipsoidOverlay(Ellipse);%显示双声速范围

            filenameZX = sprintf('zx frame=%d,VM_out=%.1f,VM_in=%.1f.png',frame, VM_out,VM_in);
            filenameZY = sprintf('zy frame=%d,VM_out=%.1f,VM_in=%.1f.png',frame, VM_out,VM_in);
            filenameXY = sprintf('xy frame=%d,VM_out=%.1f,VM_in=%.1f.png',frame, VM_out,VM_in);
            imwrite(mat2gray(squeeze(max(pa_total(:,:,:),[],1))), filenameZX);
            imwrite(mat2gray(squeeze(max(pa_total(:,:,:),[],2))), filenameZY);
            imwrite(mat2gray(squeeze(max(pa_total(:,:,:),[],3))), filenameXY);
            

        end
        
        

    case 8 %单声速相干因子法多帧覆合
        pa_total = zeros(size(Points_img(:,:,:,1)));

        % 软件中触发速度11000对应0.811°，那么软件中触发速度5000可线性计算
        delta_angle = -10000*0.811/11000; %角度°

        % 坐标旋转(角度制,顺时针为正，逆时针为负）
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
            
            dector_new=dector_new*afine_mat'; 

            % 获取并调整传感器位置，Z 坐标取负以调整坐标系统
            x_sensor=dector_new(:,1);
            y_sensor=dector_new(:,2);
            z_sensor=-dector_new(:,3); % 注意此处z轴坐标反转
            
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
            pa_img2 = rigidRegistration3D(pa_total, pa_img2);

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
            axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
            ylabel('X'); xlabel('Z'); title('XZ proj'); 
            subplot(133); imagesc(z_range, y_range, squeeze(max(pa_total(:,:,:),[], 2)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; set(gca, 'tickdir', 'out'); 
            ylabel('Y'); xlabel('Z'); title('YZ proj'); 
            subplot(132); imagesc(x_range, y_range, squeeze(max(pa_total(:,:,:),[], 3)),[imin,imax]); 
            axis equal tight; colormap gray; colorbar; 
            ylabel('Y'); xlabel('X'); title('XY proj'); set(gca, 'tickdir', 'out'); 

            filenameZX = sprintf('zx frame=%d,V_M=%.1f.png',frame, V_M);
            filenameZY = sprintf('zy frame=%d,V_M=%.1f.png',frame, V_M);
            filenameXY = sprintf('xy frame=%d,V_M=%.1f.png',frame, V_M);
            imwrite(mat2gray(squeeze(max(pa_total(:,:,:),[],1))), filenameZX);
            imwrite(mat2gray(squeeze(max(pa_total(:,:,:),[],2))), filenameZY);
            imwrite(mat2gray(squeeze(max(pa_total(:,:,:),[],3))), filenameXY);

        end

    case 9 %单声速平移覆合重建
        imgsize = size(Points_img);
        pa_img_total = zeros(imgsize(1:3));
        Nframexy = size(str_name,1)/4;

        for xframe = 1:abs(step_x):Nframex
            [datax,DAQ_time_point] = func_3D_PACT_Data_Time_Read(folder_path,str_name(1+(xframe-1)*4).name);
            pa_data = -datax;
            for yframe = 1:step_y:Nframey %设置双声速重建区域时，最好使用双声速重建复合的第一帧确定参数
                tic
                x_sensor_new = x_sensor + (xframe-1)*step_length_x;
                y_sensor_new = y_sensor - (yframe-1)*step_length_y;
                pa_data_frame = gpuArray(single(pa_data(:,:,yframe))); % [Nelemt x Nsample]
                Points_sensor_all = gpuArray(single([x_sensor_new,y_sensor_new,z_sensor]));% [Nelemt x 3]
                 
                % %对球阵传感器阵列插值
                % [pa_data_frame, Points_sensor_all] = interpolation(pa_data_frame,Points_sensor_all);
        
                % 调用 CUDA MEX 函数
                [pa_img, total_angle_weight] = SingleSpeedReconstraction_mex(Points_sensor_all, Points_img, pa_data_frame, single(fs), single(predelay), single(V_M), single(R));            % % 初始化图像缓冲区和角度权重
        
                disp(['xframe : ',num2str(xframe),'  yframe : ',num2str(yframe)]);
                toc
                % 收集GPU数据
                pa_img1 = gather(pa_img);
                total_angle_weight = gather(total_angle_weight);
                
                % 归一化并应用非线性增强
                pa_img2 = pa_img1./total_angle_weight;
                [GaussianMask, ~, ~] = generateGaussianMask({x_range, y_range}, 'Center', [(xframe-1)*step_length_x, -(yframe-1)*step_length_y], 'Sigma', 2);
                GaussianMask = GaussianMask.*ones(1,1,size(z_range,2));%z方向均匀
                pa_img3 = pa_img2.*GaussianMask;
                pa_img_total = pa_img_total + pa_img3;
                
                % 显示不同视角的图像
                pa_img_total_2 = subplus(pa_img_total);
                imin=min(pa_img_total_2,[],"all");
                imax=max(pa_img_total_2,[],"all");

                % --- 通过句柄更新数据，不会抢焦点 ---
                set(h_img9_1, 'CData', squeeze(max(pa_img2(:,:,:),[], 3)));
                set(h_img9_2, 'CData', squeeze(max(GaussianMask(:,:,:),[], 3)));
                set(h_img9_3, 'CData', squeeze(max(pa_img3(:,:,:),[], 3)));
                
                % 更新 Figure 10 的数据
                set(h_img10_1, 'CData', squeeze(max(pa_img_total_2(:,:,:),[],1)));
                set(h_img10_1.Parent, 'CLim', [imin, imax]); 
                set(h_img10_2, 'CData', squeeze(max(pa_img_total_2(:,:,:),[],2)));
                set(h_img10_2.Parent, 'CLim', [imin, imax]); 
                set(h_img10_3, 'CData', squeeze(max(pa_img_total_2(:,:,:),[],3)));
                set(h_img10_3.Parent, 'CLim', [imin, imax]); 
                % drawEllipsoidOverlay(Ellipse);%显示双声速范围
                
                % 刷新绘图，limitrate 限制刷新频率
                % 既能看到动图，又不抢焦点。
                drawnow limitrate;

                filenameZX = sprintf('step=%.1fmm zx xframe=%d, yframe=%d.png',step_x, xframe, yframe);
                filenameZY = sprintf('step=%.1fmm zy xframe=%d, yframe=%d.png',step_x, xframe, yframe);
                filenameXY = sprintf('step=%.1fmm xy xframe=%d, yframe=%d.png',step_x, xframe, yframe);
                imwrite(mat2gray(squeeze(max(pa_img_total_2(:,:,:),[],1))), filenameZX);
                imwrite(mat2gray(squeeze(max(pa_img_total_2(:,:,:),[],2))), filenameZY);
                imwrite(mat2gray(squeeze(max(pa_img_total_2(:,:,:),[],3))), filenameXY);
    
            end
        end
    
    case 10 %单/双声速旋转平移覆合
        %% 子程序对应采集数据说明：
        % 为了保证不同数据的方向相同，程序固定第一帧为覆合模板，因此在采集数据时应当先采集再旋转
        % 采集数据时需保证当前帧不为呼吸帧
        % 当前程序设定采集帧率为10Hz，程序中包含未旋转帧过滤程序，默认过滤从开始采集后1.5s内的数据，确保覆合数据均为旋转数据
        imgsize = size(Points_img);
        pa_img_total = zeros(imgsize(1:3));
        Ndata = size(str_name,1)/4;%采集数据组数
        Nframex = 4; %x方向扫描次数
        Nframey = Ndata/Nframex; %y方向扫描次数
        
        for xframe = 1:step_x:Nframex

            for yframe = 1:step_y:Nframey 
                frame_idx = 1+((xframe-1)*step_x*Nframey+(yframe-1)*step_y)*4;
                [datax,DAQ_time_point] = func_3D_PACT_Data_Time_Read(folder_path,str_name(frame_idx).name);
                pa_data = -datax(:,:,1:2:end);
                figure(1);imagesc(pa_data(:,:,1),[-100,100]);

                x_sensor_new = x_sensor + (xframe-1)*step_length_x;
                y_sensor_new = y_sensor - (yframe-1)*step_length_y;
                pa_data_frame = gpuArray(single(pa_data(:,:,yframe))); % [Nelemt x Nsample]
                dector_new = gpuArray(single([x_sensor_new,y_sensor_new,z_sensor,z_sensor*0+1]));% [Nelemt x 3]
        %%
                pa_total = zeros(size(Points_img(:,:,:,1)),'single');
        
                corr_mat = zeros(Nframe,Nframe);
                % 产生静态帧-门控
                common = mean(pa_data,3);
                tic
        
                for frx = 1:Nframe
                    for fry = frx:Nframe
                        corr_res = corrcoef(pa_data(:,2401:3200,frx),pa_data(:,2401:3200,fry));
                        corr_mat(frx,fry) = corr_res(1,2);%对称相似度矩阵
                    end
                end
                corr_mat = corr_mat+corr_mat';%补全简化计算的部分
                corr_line = mean(corr_mat,1);
                corr_line = corr_line/max(corr_line(3:end));%避免静止帧相关性太强导致归一化后系数偏小
                corr_line(1) = 1; %先采集后旋转时，强制首帧参考
                corr_line(2:3) = 0; %先采集后旋转时，去除未旋转的数据
                static_frames = 1:Nframe;
                top_vals = maxk(corr_line, 20);%找出最大的20个值，并按降序排列
                Similarity_threshold = top_vals(end);%动态调整覆合时所用的相似度阈值，避免不同组数据相似度波动导致覆合帧数不同
                static_frames = static_frames(find(corr_line>=Similarity_threshold));
                figure,plot(corr_line,'b'),hold on
                for isf = static_frames
                    plot(isf,corr_line(isf),'*r'),hold on
                end
                hold off;
                
                
                % 软件中触发速度11000对应0.811°，那么软件中触发速度10000可线性计算-改
                delta_angle = -5500*0.816/11000; %角度°
                static_Nframe = size(static_frames,2);
                
                firstframe_flag = 1; %首帧标识 1：是首帧，0：非首帧

                for frame = 1:1:static_Nframe %注意这里的step*delta_angle为相邻复合角
            
                    tic
                    % 坐标旋转(角度制,顺时针为正，逆时针为负）
                    theta_x = 0; % 以x轴为中心旋转 
                    theta_y = 0; % 以y轴为中心旋转 
                    theta_z = (static_frames(frame)-static_frames(1))*delta_angle; % 以z轴为中心旋转 
                    
                    % 定义旋转中心坐标（匹配物理位移）
                    Rotate_center_x = (xframe-1)*step_length_x; %每次平移修改旋转中心
                    Rotate_center_y = -(yframe-1)*step_length_y; 
                    Rotate_center_z = 0;

                    %坐标整体平移(这里的坐标平移会影响椭球中心选取)
                    trans_x = 0;
                    trans_y = 0;
                    trans_z = 0;
                    
                    rotate_x_mat = [1 0 0 0;0 cosd(theta_x) -sind(theta_x),0;0 sind(theta_x) cosd(theta_x) 0;0 0 0 1];
                    rotate_y_mat = [cosd(theta_y) 0 -sind(theta_y) 0;0 1 0,0;sind(theta_y) 0 cosd(theta_y) 0;0 0 0 1];
                    rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0;sind(theta_z) cosd(theta_z) 0,0;0 0 1 0;0 0 0 1];
                    trans_mat = [1 0 0 trans_x;0 1 0 trans_y;0 0 1 trans_z;0 0 0 1];
                    
                    R_total = rotate_x_mat * rotate_y_mat * rotate_z_mat;% 合并旋转矩阵 (先Z后Y后X)
                    
                    % 移向新原点的矩阵 (Translate to new Origin)
                    T_to_origin = [1 0 0 -Rotate_center_x; 
                                   0 1 0 -Rotate_center_y; 
                                   0 0 1 -Rotate_center_z; 
                                   0 0 0 1];
                    
                    % 移回原位的矩阵 (Translate Back)
                    T_back = [1 0 0 Rotate_center_x; 
                              0 1 0 Rotate_center_y; 
                              0 0 1 Rotate_center_z; 
                              0 0 0 1];

                    afine_mat = trans_mat * T_back * R_total * T_to_origin;%先平移修改旋转原点，再旋转，再将原点改为原来的值，最后整体位移
                    dector_corr=dector_new*afine_mat'; 
        
                    pa_data_frame = gpuArray(single(pa_data(:,:,static_frames(frame)))); % [Nelemt x Nsample]
                    Points_sensor_all = gpuArray(single(dector_corr(:,1:3)));% [Nelemt x 3]
                    
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
                    [GaussianMask, ~, ~] = generateGaussianMask({x_range, y_range}, 'Center', [(xframe-1)*step_length_x, -(yframe-1)*step_length_y], 'Sigma', 15);
                    GaussianMask = GaussianMask.*ones(1,1,size(z_range,2));%z方向均匀
                    pa_img3 = pa_img2.*GaussianMask;
                    % pa_img3 = pa_img2.*(2-GaussianMask); %取消高斯滤波并进行高斯增强
        
                    if firstframe_flag == 0 %判断是否为首帧
                        %图像配准
                        % pa_img3 = rigidRegistration3D(pa_ref, pa_img3);
        
                    else
                        pa_ref = pa_img3;%固定首帧作为参考帧
            
                    end
                    pa_total = pa_total+pa_img3;
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

                    filenameZX = sprintf('step=%.1fmm zx xframe=%d, yframe=%d, frame=%d.png',step_x, xframe, yframe, frame);
                    filenameZY = sprintf('step=%.1fmm zy xframe=%d, yframe=%d, frame=%d.png',step_x, xframe, yframe, frame);
                    filenameXY = sprintf('step=%.1fmm xy xframe=%d, yframe=%d, frame=%d.png',step_x, xframe, yframe, frame);
                    imwrite(mat2gray(squeeze(max(pa_total(:,:,:),[],1))), filenameZX);
                    imwrite(mat2gray(squeeze(max(pa_total(:,:,:),[],2))), filenameZY);
                    imwrite(mat2gray(squeeze(max(pa_total(:,:,:),[],3))), filenameXY);

                end % frame end

                toc
                pa_img_total = pa_img_total + pa_total;
                pa_img_total_2 = subplus(pa_img_total);
                imin=min(pa_img_total_2,[],"all");
                imax=max(pa_img_total_2,[],"all");

                % 更新 Figure 10 的数据
                set(h_img10_1, 'CData', squeeze(max(pa_img_total(:,:,:),[],1)));
                set(h_img10_1.Parent, 'CLim', [imin, imax]); 
                set(h_img10_2, 'CData', squeeze(max(pa_img_total(:,:,:),[],3)));
                set(h_img10_2.Parent, 'CLim', [imin, imax]); 
                set(h_img10_3, 'CData', squeeze(max(pa_img_total(:,:,:),[],2)));
                set(h_img10_3.Parent, 'CLim', [imin, imax]); 
                % drawEllipsoidOverlay(Ellipse);%显示双声速范围
                
                drawnow limitrate;

                filenameZX = sprintf('zx frame=%d,V_M=%.1f.png',frame, V_M);
                filenameZY = sprintf('zy frame=%d,V_M=%.1f.png',frame, V_M);
                filenameXY = sprintf('xy frame=%d,V_M=%.1f.png',frame, V_M);
                imwrite(mat2gray(squeeze(max(pa_total(:,:,:),[],1))), filenameZX);
                imwrite(mat2gray(squeeze(max(pa_total(:,:,:),[],2))), filenameZY);
                imwrite(mat2gray(squeeze(max(pa_total(:,:,:),[],3))), filenameXY);
                                
                filenameZX = sprintf('step=%.1fmm zx xframe=%d, yframe=%d.png',step_x, xframe, yframe);
                filenameZY = sprintf('step=%.1fmm zy xframe=%d, yframe=%d.png',step_x, xframe, yframe);
                filenameXY = sprintf('step=%.1fmm xy xframe=%d, yframe=%d.png',step_x, xframe, yframe);
                imwrite(mat2gray(squeeze(max(pa_img_total_2(:,:,:),[],1))), filenameZX);
                imwrite(mat2gray(squeeze(max(pa_img_total_2(:,:,:),[],2))), filenameZY);
                imwrite(mat2gray(squeeze(max(pa_img_total_2(:,:,:),[],3))), filenameXY);
            end %yframe end
        end %xframe end
       save('pa_img_total_2_95%.mat', 'pa_img_total_2');
    otherwise

        disp('Error: Undefined reconstruct mode!');

end


toc % 结束计时


