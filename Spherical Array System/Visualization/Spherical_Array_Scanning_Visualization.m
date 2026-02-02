%% ********************** 步进-旋转复合扫描可视化 ***********************
% 本程序用于展示扫描复合数据采集流程
% 
% 逻辑：到达位置 -> 旋转采集 -> 移动到下一点 (虚线) -> 再次旋转
%
% Author：Xiali Gao  Version：1.0  2026.2.2
% **************************************************************************

% close all; 
clear; clc;

% --- 1. 参数设置 ---
R_array = 100;          % 阵列半径 (mm)
N_elements = 1024;       % 换能器数量
rot_steps = 20;         % 每个点旋转的角度步数 (例如总转360度)
rot_angle_inc = 0.8;     % 每步转0.8度

% 定义三维扫描网格 (3层, 每层 3x3)
scan_x = 0:80:160;       
scan_y = 0:80:160;       
scan_z = [0, -50, -100]; % 不同深度

% --- 2. 生成初始阵列 (Fibonacci) ---
phi = pi * (3 - sqrt(5)); 
i = 0:N_elements-1;
z_val = (i / (N_elements - 1)); 
r_val = sqrt(1 - z_val.^2);
theta = phi * i;
x0 = R_array * r_val .* cos(theta);
y0 = R_array * r_val .* sin(theta);
z0 = -R_array * z_val; 

% --- 3. 环境初始化 ---
fig = figure('Color', [0.05 0.05 0.08], 'Name', 'Step-and-Scan Visualization');
ax = axes('Color', 'none', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
hold on; axis equal; grid on;
view(40, 35);
axis([-110 270 -110 270 -210 10]);
% pause();

% 颜色准备：为不同 Z 深度准备颜色
depth_colors = lines(length(scan_z)); 
colormap(winter);

% 初始化阵列图形对象 (使用散点)
hArray = scatter3(x0, y0, z0, 35, z0, 'filled', 'MarkerFaceAlpha', 0.7);
camlight headlight; lighting gouraud;

% --- 4. 核心扫描循环 ---
title('Sequential Step-and-Scan Protocol', 'Color', 'w', 'FontSize', 13);
last_pos = [0, 0, 0]; % 记录上一个位置用于画虚线

for zi = 1:length(scan_z)
    z_curr = scan_z(zi);
    c_curr = depth_colors(zi, :); % 当前深度的轨迹颜色
    
    for yi = 1:length(scan_y)
        % 蛇形扫描逻辑
        curr_x_range = scan_x;
        if mod(yi, 2) == 0, curr_x_range = fliplr(scan_x); end
        
        for xi = 1:length(curr_x_range)
            x_curr = curr_x_range(xi);
            y_curr = scan_y(yi);
            target_pos = [x_curr, y_curr, z_curr];
            
            % --- A. 平移过程 (不旋转) ---
            % 绘制从上一点到当前点的虚线路径
            if ~(zi==1 && yi==1 && xi==1)
                line([last_pos(1) target_pos(1)], ...
                     [last_pos(2) target_pos(2)], ...
                     [last_pos(3) target_pos(3)], ...
                     'Color', c_curr, 'LineStyle', '--', 'LineWidth', 1.5);
            end
            
            % 更新阵列位置
            set(hArray, 'XData', x0 + x_curr, 'YData', y0 + y_curr, 'ZData', z0 + z_curr);
            last_pos = target_pos;
            drawnow; pause(0.1);
            
            % --- B. 采集位置标记 ---
            % 在旋转开始前，打一个醒目的点表示这里是采集站
            plot3(x_curr, y_curr, z_curr, 'o', 'MarkerSize', 8, ...
                  'MarkerFaceColor', c_curr, 'MarkerEdgeColor', 'w');
            text(x_curr+5, y_curr+5, z_curr, sprintf('Pos(%d,%d,%d)', x_curr, y_curr, z_curr), ...
                 'Color', 'w', 'FontSize', 12);

            % --- C. 旋转采集过程 (不平移) ---
            pause(0.2)
            for r = 1:rot_steps
                angle = r * rot_angle_inc;
                Rz = [cosd(angle) -sind(angle) 0; sind(angle) cosd(angle) 0; 0 0 1];
                
                % 计算旋转坐标并加上当前平移量
                rotated_coords = (Rz * [x0; y0; z0])';
                set(hArray, 'XData', rotated_coords(:,1) + x_curr, ...
                            'YData', rotated_coords(:,2) + y_curr, ...
                            'ZData', rotated_coords(:,3) + z_curr);
                
                % 视觉提示：正在旋转采集
                title(['Acquiring at Z = ', num2str(z_curr), ' mm (Rotating: ', num2str(angle), '°)'], 'Color', c_curr);
                drawnow limitrate;
                pause(0.2)
            end
            pause(0.2)
        end
    end
end

% 最终视角美化
title('Multi-layer 3D Scanning Complete', 'Color', 'g');