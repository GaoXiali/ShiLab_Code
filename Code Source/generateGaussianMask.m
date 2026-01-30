function [field, X, Y] = generateGaussianMask(gridDef, varargin)
% GENERATEGAUSSIANMASK 生成指定坐标系下的二维高斯分布
%
% 输入 gridDef 可以是:
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

% %%example
% %% 场景设置：物理尺寸
% % 定义 x, y 方向的物理坐标向量 (例如单位: mm)
% x_axis = linspace(-10, 10, 200)'; % x方向范围
% y_axis = linspace(-5, 5, 100)';   % y方向范围
% 
% % 1. 正常情况：高斯在中心
% figure;
% subplot(1,3,1);
% [field1, ~, ~] = generateGaussianMask({x_axis, y_axis}, ...
%     'Center', [0, 0], 'Sigma', 2);
% imagesc(x_axis, y_axis, field1);
% axis image; title('中心对齐');
% colorbar;
% 
% % 2. 偏移情况：高斯中心偏移到 (8, 3)，大部分在图像边缘
% subplot(1,3,2);
% [field2, ~, ~] = generateGaussianMask({x_axis, y_axis}, ...
%     'Center', [8, 3], 'Sigma', 3);
% imagesc(x_axis, y_axis, field2);
% axis image; title('偏移 (部分在视场外)');
% colorbar;
% 
% % 3. 极端情况：高斯中心完全在视场外 (12, 0)，只看到一点尾巴
% subplot(1,3,3);
% [field3, ~, ~] = generateGaussianMask({x_axis, y_axis}, ...
%     'Center', [12, 0], 'Sigma', 3);
% imagesc(x_axis, y_axis, field3);
% axis image; title('中心在视场外');
% colorbar;

    %% 1. 解析网格定义 (Grid Definition)
    if iscell(gridDef) && numel(gridDef) == 2
        % --- 物理坐标模式 ---
        x_vec = gridDef{1};
        y_vec = gridDef{2};
        % 确保是列向量或行向量均可，后续统一处理
        x_vec = x_vec(:)';
        y_vec = y_vec(:)';
    elseif isnumeric(gridDef) && numel(gridDef) == 2
        % --- 像素模式 ---
        rows = gridDef(1);
        cols = gridDef(2);
        x_vec = 1:cols;
        y_vec = 1:rows;
    else
        error('Input format error: gridDef must be [rows, cols] or {x_vec, y_vec}');
    end

    % 生成网格 (Meshgrid)
    % 注意: meshgrid 的第一个参数对应列(x), 第二个参数对应行(y)
    [X, Y] = meshgrid(x_vec, y_vec);

    %% 2. 解析其他参数
    p = inputParser;
    
    % 默认中心: 视场的几何中心
    defaultCenter = [mean(x_vec), mean(y_vec)];
    
    % 默认 Sigma: 视场最短边长的 1/4
    spanX = max(x_vec) - min(x_vec);
    spanY = max(y_vec) - min(y_vec);
    defaultSigma = min(spanX, spanY) / 4;
    if defaultSigma == 0, defaultSigma = 1; end % 防止单点情况除以0

    addParameter(p, 'Center', defaultCenter, @(x) validateattributes(x, {'numeric'}, {'vector', 'numel', 2}));
    addParameter(p, 'Sigma', defaultSigma, @(x) validateattributes(x, {'numeric'}, {'vector'}));
    addParameter(p, 'Angle', 0, @(x) isscalar(x));
    addParameter(p, 'Norm', 'max', @(x) any(validatestring(x, {'max', 'sum'})));
    
    parse(p, varargin{:});
    
    xc = p.Results.Center(1);
    yc = p.Results.Center(2);
    sigma = p.Results.Sigma;
    angle = p.Results.Angle;
    normType = p.Results.Norm;

    % 处理各向异性 Sigma
    if isscalar(sigma)
        sig_x = sigma;
        sig_y = sigma;
    else
        sig_x = sigma(1);
        sig_y = sigma(2);
    end

    %% 3. 坐标变换 (平移 + 旋转)
    % 将网格坐标平移到以高斯中心为原点
    X_shifted = X - xc;
    Y_shifted = Y - yc;

    % 旋转坐标系 (逆时针旋转)
    if angle ~= 0
        theta = deg2rad(angle);
        % 这里使用的是坐标旋转公式
        X_rot = X_shifted * cos(theta) - Y_shifted * sin(theta);
        Y_rot = X_shifted * sin(theta) + Y_shifted * cos(theta);
    else
        X_rot = X_shifted;
        Y_rot = Y_shifted;
    end

    %% 4. 计算高斯分布
    % 核心公式
    exponent = -((X_rot.^2) / (2 * sig_x^2) + (Y_rot.^2) / (2 * sig_y^2));
    field = exp(exponent);

    %% 5. 归一化处理
    if strcmp(normType, 'max')
        % 峰值归一化: 无论中心是否在视场内，假设无限远处的理论峰值为1
        % 注意: 这里通常不需要除以 max(field(:))，因为 exp(0)=1。
        % 如果中心在视场外，max(field) 会小于1，这正是我们要的“局部”效果。
        % 所以对于 'max' 模式，直接返回 exp 计算结果即可保持物理意义。
        % 如果你强制要求视场内现有最大值为1，请取消下面注释：
        % field = field / max(field(:)); 
    elseif strcmp(normType, 'sum')
        field = field / sum(field(:));
    end
end

