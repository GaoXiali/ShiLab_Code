%% **************************斐波那契阵列展示*******************************
%
% 本代码用于半球阵光声成像系统中，按照斐波那契阵列排布的换能器位置可视化
% Author：Xiali Gao  Version：1.0  2026.1.28
%
%**************************************************************************
close all; clear; clc;
addpath 'D:\GXL\DataRead';

%% ========================= 参数设置 =========================
% 可视化参数
num_rotations = 200;          % 旋转帧数
rotation_step = 0.8;         % 每帧旋转角度(度)
pause_time = 0.05;           % 动画帧间隔(秒)
point_size = 60;             % 探测器点大小
sphere_radius = 100;         % 参考半球半径
sphere_alpha = 0.15;         % 半球透明度

% 颜色方案 (可选: 'ocean', 'sunset', 'neon', 'scientific')
color_scheme = 'sunset';

%% ========================= 加载数据 =========================
dector = load('coordinate.txt');
x = dector(:,1);
y = dector(:,2);
z = -dector(:,3);
num_points = size(dector, 1);

%% ========================= 颜色配置 =========================
switch color_scheme
    case 'ocean'
        cmap = [linspace(0.1, 0.2, 256)', linspace(0.3, 0.8, 256)', linspace(0.6, 1.0, 256)'];
        bg_color = [0.02, 0.05, 0.1];
        sphere_color = [0.3, 0.6, 0.9];
        grid_color = [0.2, 0.4, 0.6];
    case 'sunset'
        cmap = [linspace(1, 1, 256)', linspace(0.2, 0.8, 256)', linspace(0.1, 0.4, 256)'];
        bg_color = [0.05, 0.02, 0.08];
        sphere_color = [0.9, 0.5, 0.3];
        grid_color = [0.6, 0.3, 0.4];
    case 'neon'
        cmap = [linspace(0, 1, 256)', linspace(1, 0, 256)', linspace(0.5, 1, 256)'];
        bg_color = [0.02, 0.02, 0.05];
        sphere_color = [0.5, 0.2, 0.8];
        grid_color = [0.4, 0.2, 0.6];
    case 'scientific'
        cmap = parula(256);
        bg_color = [1, 1, 1];
        sphere_color = [0.7, 0.7, 0.7];
        grid_color = [0.5, 0.5, 0.5];
end

%% ========================= 创建图形窗口 =========================
fig = figure('Name', '斐波那契阵列可视化', ...
             'Color', bg_color, ...
             'Position', [100, 100, 1200, 900], ...
             'Renderer', 'opengl');

ax = axes('Parent', fig);
hold(ax, 'on');

%% ========================= 绘制参考半球 =========================
[sphere_x, sphere_y, sphere_z] = sphere(50);
sphere_x = sphere_x * sphere_radius;
sphere_y = sphere_y * sphere_radius;
sphere_z = sphere_z * sphere_radius;

% 只保留下半球
sphere_z(sphere_z > 0) = NaN;

surf(ax, sphere_x, sphere_y, sphere_z, ...
     'FaceColor', sphere_color, ...
     'FaceAlpha', sphere_alpha, ...
     'EdgeColor', 'none', ...
     'FaceLighting', 'gouraud');

%% ========================= 绘制坐标轴参考线 =========================
axis_length = sphere_radius * 1.3;
line_width = 1.5;

% X轴 (红色)
plot3(ax, [-axis_length, axis_length], [0, 0], [0, 0], ...
      'Color', [0.9, 0.3, 0.3, 0.6], 'LineWidth', line_width);
% Y轴 (绿色)
plot3(ax, [0, 0], [-axis_length, axis_length], [0, 0], ...
      'Color', [0.3, 0.9, 0.3, 0.6], 'LineWidth', line_width);
% Z轴 (蓝色)
plot3(ax, [0, 0], [0, 0], [-axis_length*0.9, axis_length*0.2], ...
      'Color', [0.3, 0.3, 0.9, 0.6], 'LineWidth', line_width);

%% ========================= 绘制底部圆环 =========================
theta_ring = linspace(0, 2*pi, 100);
ring_r = sphere_radius;
ring_x = ring_r * cos(theta_ring);
ring_y = ring_r * sin(theta_ring);
ring_z = zeros(size(theta_ring));
plot3(ax, ring_x, ring_y, ring_z, 'Color', [grid_color, 0.5], 'LineWidth', 1.5);

%% ========================= 初始探测器点绘制 =========================
% 根据深度计算颜色
z_normalized = (z - min(z)) / (max(z) - min(z));
point_colors = interp1(linspace(0, 1, size(cmap, 1)), cmap, z_normalized);

% 初始点 (半透明)
scatter3(ax, x, y, z, point_size * 0.5, point_colors, 'filled', ...
         'MarkerFaceAlpha', 0.3, 'MarkerEdgeColor', 'none');

% 当前动态点的句柄
h_points = scatter3(ax, x, y, z, point_size, point_colors, 'filled', ...
                    'MarkerEdgeColor', [1, 1, 1], 'MarkerEdgeAlpha', 0.3, ...
                    'LineWidth', 0.5);

%% ========================= 添加轨迹线 (可选) =========================
% 选取几个代表性的点绘制轨迹
trail_indices = round(linspace(1, num_points, min(8, num_points)));
trail_handles = gobjects(length(trail_indices), 1);

for idx = 1:length(trail_indices)
    trail_handles(idx) = plot3(ax, NaN, NaN, NaN, ...
                               'Color', [point_colors(trail_indices(idx), :), 0.4], ...
                               'LineWidth', 1.5);
end

%% ========================= 设置视图和光照 =========================
view(ax, 35, 25);
axis(ax, 'equal');
xlim(ax, [-axis_length, axis_length]);
ylim(ax, [-axis_length, axis_length]);
zlim(ax, [-120, 30]);

% 添加光照
light(ax, 'Position', [1, 0.5, 1], 'Style', 'infinite');
light(ax, 'Position', [-1, -0.5, 0.5], 'Style', 'infinite', 'Color', [0.3, 0.3, 0.4]);
lighting(ax, 'gouraud');

% 设置坐标轴样式
set(ax, 'Color', 'none', ...
        'XColor', grid_color, 'YColor', grid_color, 'ZColor', grid_color, ...
        'GridColor', grid_color, 'GridAlpha', 0.3, ...
        'FontSize', 10, 'FontName', 'Arial');
grid(ax, 'on');
box(ax, 'on');

xlabel(ax, 'X (mm)', 'Color', grid_color, 'FontWeight', 'bold');
ylabel(ax, 'Y (mm)', 'Color', grid_color, 'FontWeight', 'bold');
zlabel(ax, 'Z (mm)', 'Color', grid_color, 'FontWeight', 'bold');

%% ========================= 添加标题和信息 =========================
title_color = [0.9, 0.9, 0.9];
if strcmp(color_scheme, 'scientific')
    title_color = [0.1, 0.1, 0.1];
end

title(ax, sprintf('半球阵光声成像系统 - 斐波那契阵列排布\n探测器数量: %d', num_points), ...
      'Color', title_color, 'FontSize', 14, 'FontWeight', 'bold');

% 添加信息文本
info_text = annotation('textbox', [0.02, 0.02, 0.25, 0.08], ...
                       'String', sprintf('帧: 0/%d\n旋转角度: 0.0°', num_rotations), ...
                       'Color', title_color, 'FontSize', 10, ...
                       'EdgeColor', 'none', 'BackgroundColor', 'none');

%% ========================= 旋转动画 =========================
% 存储轨迹点
trail_x = cell(length(trail_indices), 1);
trail_y = cell(length(trail_indices), 1);
trail_z = cell(length(trail_indices), 1);

for i = 1:num_rotations
    % 坐标旋转参数
    theta_x = 0;
    theta_y = 0;
    theta_z = rotation_step * i;
    
    % 构建旋转矩阵
    rotate_x_mat = [1 0 0 0; 0 cosd(theta_x) -sind(theta_x) 0; 0 sind(theta_x) cosd(theta_x) 0; 0 0 0 1];
    rotate_y_mat = [cosd(theta_y) 0 -sind(theta_y) 0; 0 1 0 0; sind(theta_y) 0 cosd(theta_y) 0; 0 0 0 1];
    rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0; sind(theta_z) cosd(theta_z) 0 0; 0 0 1 0; 0 0 0 1];
    
    affine_mat = rotate_x_mat * rotate_y_mat * rotate_z_mat;
    
    % 应用变换
    dector_homo = [dector, ones(num_points, 1)];
    dector_new = dector_homo * affine_mat';
    
    x_new = dector_new(:, 1);
    y_new = dector_new(:, 2);
    z_new = -dector_new(:, 3);
        
    % 更新探测器点位置
    set(h_points, 'XData', x_new, 'YData', y_new, 'ZData', z_new);
    
    % 更新轨迹线
    for idx = 1:length(trail_indices)
        pt_idx = trail_indices(idx);
        trail_x{idx} = [trail_x{idx}, x_new(pt_idx)];
        trail_y{idx} = [trail_y{idx}, y_new(pt_idx)];
        trail_z{idx} = [trail_z{idx}, z_new(pt_idx)];
        
        % 只保留最近的N个点
        max_trail = 15;
        if length(trail_x{idx}) > max_trail
            trail_x{idx} = trail_x{idx}(end-max_trail+1:end);
            trail_y{idx} = trail_y{idx}(end-max_trail+1:end);
            trail_z{idx} = trail_z{idx}(end-max_trail+1:end);
        end
        
        set(trail_handles(idx), 'XData', trail_x{idx}, ...
                                'YData', trail_y{idx}, ...
                                'ZData', trail_z{idx});
    end
    
    % 更新信息文本
    set(info_text, 'String', sprintf('帧: %d/%d\n旋转角度: %.1f°', i, num_rotations, theta_z));
    
    % 可选: 轻微旋转视角
    % current_view = get(ax, 'View');
    % set(ax, 'View', [current_view(1) + 0.5, current_view(2)]);
    
    drawnow;
    pause(pause_time);
end

%% ========================= 最终状态 =========================
% 添加最终状态提示
set(info_text, 'String', sprintf('动画完成\n总旋转角度: %.1f°', rotation_step * num_rotations));

% 可选: 保存最终图像
% saveas(fig, 'fibonacci_array_final.png');
% exportgraphics(fig, 'fibonacci_array_final.pdf', 'ContentType', 'vector');

disp('可视化完成！');
disp(['探测器数量: ', num2str(num_points)]);
disp(['总旋转角度: ', num2str(rotation_step * num_rotations), '°']);
