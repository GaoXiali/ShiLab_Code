% 1. 准备示例数据 (假设已知圆心为 [2, 3], 半径为 5)
% 在实际使用中，请替换为你的坐标矩阵 [x, y]
% theta_true = linspace(0, 1.5*pi, 10)'; % 模拟转过1.5pi的弧度
% X_raw = 2 + 5*cos(theta_true) + 0.05*randn(size(theta_true)); % 加一点噪声
% Y_raw = 3 + 5*sin(theta_true) + 0.05*randn(size(theta_true));
% points = [3.65, 10.65;5.15,9.9;6.55,8.95;7.8,7.9;8.9,6.7;9.75,5.35;10.5,3.9];
points = [-1.04,11.38;0.4,11.40;1.86,11.16;3.32,10.74;4.68,10.22;5.96,9.46;7.2,8.46;8.2,7.5;9.12,6.42;9.88,5.06;10.52,3.74];
% points = [-2.3,-5.5;-3,-5.3;-3.65,-4.9;-4.2,-4.3;];


%% 2. 拟合圆心 (xc, yc) 和半径 R
% 圆的方程: (x-xc)^2 + (y-yc)^2 = R^2
% 展开为线性方程形式: 2*x*xc + 2*y*yc + (R^2 - xc^2 - yc^2) = x^2 + y^2
% 令常数项 C = R^2 - xc^2 - yc^2
x = points(:, 1);
y = points(:, 2);

% 构建超定方程组 A * [xc; yc; C] = B
A = [2*x, 2*y, ones(size(x))];
B = x.^2 + y.^2;

% 使用反斜杠算子求解最小二乘解
coeffs = A \ B;
xc = coeffs(1);
yc = coeffs(2);
C = coeffs(3);
R = sqrt(C + xc^2 + yc^2);

fprintf('拟合圆心坐标: (%.4f, %.4f)\n', xc, yc);
fprintf('拟合半径 R: %.4f\n', R);

%% 3. 计算每个点相对于圆心的极角及点间的转角
% 将坐标平移，使圆心位于原点
x_rel = x - xc;
y_rel = y - yc;

% 计算每个点相对于圆心的绝对角度 (弧度，范围 -pi 到 pi)
angles = atan2(y_rel, x_rel); 

% 计算相邻点之间的转角 (Delta Theta)
% 使用 unwrap 处理从 pi 跳变到 -pi 的情况
delta_angles = diff(unwrap(angles)); 

% 转换为角度单位 (可选)
delta_angles_deg = rad2deg(delta_angles);

%% 4. 结果展示
disp('相邻点之间的转角 (度):');
disp(delta_angles_deg);

% 可视化
figure;
plot(x, y, 'ro', 'DisplayName', '原始采样点'); hold on;
viscircles([xc, yc], R, 'Color', 'b', 'LineStyle', '--');
plot(xc, yc, 'bx', 'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', '拟合圆心');
axis equal; grid on;
legend;
title('圆心拟合与采样点分布');

A = [   -6.9904
   -7.1788
   -7.3859
   -7.0656
   -7.2194
   -7.7454
   -6.7474
   -6.8841
   -7.5633
   -7.1258];
mean(A-0.811)
