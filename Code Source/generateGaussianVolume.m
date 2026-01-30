function [vol, X, Y] = generateGaussianVolume(gridDef, varargin)
% GENERATEGAUSSIANVOLUME 生成二维或三维高斯分布场
%
% 用法:
%   vol = generateGaussianVolume({x_vec, y_vec}, 'Center', [0,0], 'Sigma', 5)
%   vol = generateGaussianVolume({x_vec, y_vec}, ..., 'ZProfile', z_vec)
%
% 输入:
% gridDef 可以是:
%   1. [rows, cols] - 像素模式 (默认 0 到 rows/cols, 单位像素)
%   2. {x_vec, y_vec} - 物理坐标模式 (指定 x 和 y 的坐标向量) 行向量或列向量均可
%
% 可选参数 (Name-Value):
%   'Center' - [xc, yc] 高斯中心坐标. (注意: 对应 gridDef 的单位)
%              如果未指定，默认为视场(FOV)的几何中心.
%   'Sigma'  - 高斯宽度 (标量或 [sig_x, sig_y]). 单位同 gridDef. sigma = 0.425*FWHM
%   'Angle'  - 旋转角度 (度).
%   'Norm'   - 'max' (峰值1) 或 'sum' (积分1, 近似).
%
% 输出:
%   field - 计算出的二维高斯矩阵
%   X, Y  - 对应的网格坐标矩阵 (方便 surf/imagesc 使用)
%   gridDef - [rows, cols] (像素模式) 或 {x_vec, y_vec} (物理坐标模式)
%
%   'ZProfile' - 一维向量. 用于在第三维(深度/Z轴)上的强度分布.
%                可以是衰减曲线，也可以是另一个高斯分布.


%% example
% % 1. 定义空间 (物理尺寸 mm)
% x = linspace(-10, 10, 100);
% y = linspace(-10, 10, 100);
% z = linspace(0, 20, 50); % 深度 0 到 20mm
% 
% % 2. 定义深度上的光衰减 (Beer-Lambert Law)
% % 假设穿透深度为 5mm (1/e)
% mu_eff = 1/5; 
% z_attenuation = exp(-mu_eff * z);
% 
% % 3. 一键生成 3D 初始压力场 P0
% % 这是一个椭圆高斯光斑 (长轴在x方向)，且沿z轴指数衰减
% P0 = generateGaussianVolume({x, y}, ...
%     'Center', [0, 0], ...     % XY中心
%     'Sigma', [3, 1.5], ...    % 椭圆光斑：x宽，y窄
%     'Angle', 30, ...          % 旋转30度
%     'ZProfile', z_attenuation); % 赋予深度信息
% 
% % 4. 检查结果
% sliceViewer(P0);
% figure; imagesc(squeeze(P0(50,:,:))'); title('侧视图 (Y-Z plane)'); xlabel('Y'); ylabel('Z');


    %% 1. 提取 ZProfile 参数，其余参数传给解析器
    % 我们先检查有没有 'ZProfile'，因为它不属于二维计算的部分
    zProfile = [];
    keepArgs = {};
    
    k = 1;
    while k <= length(varargin)
        if strcmpi(varargin{k}, 'ZProfile')
            zProfile = varargin{k+1};
            k = k + 2;
        else
            keepArgs{end+1} = varargin{k}; %#ok<AGROW>
            if k+1 <= length(varargin) && ~ischar(varargin{k+1}) 
                 % 简单的参数值判断，防止跳过
            end
            k = k + 1;
        end
    end
  
    [field2D, X, Y] = generateGaussianMask(gridDef, keepArgs{:});

    %% 3. 处理三维扩展 
    if isempty(zProfile)
        % 如果没有指定 Z 方向分布，直接返回二维结果
        vol = field2D;
        disp('Output: 2D Matrix');
    else
        % 确保 zProfile 是列向量，并且非空
        zProfile = zProfile(:);
        
        if isempty(zProfile)
            error('ZProfile cannot be empty.');
        end
        
        % 利用维度广播 (Implicit Expansion) 生成三维数据
        % field2D 是 (Nx, Ny)
        % reshape(zProfile, 1, 1, Nz) 把向量竖在第三维
        vol = field2D .* reshape(zProfile, 1, 1, []);
        
        fprintf('Output: 3D Matrix [%d x %d x %d]\n', ...
            size(vol, 1), size(vol, 2), size(vol, 3));
    end
end

