function drawEllipsoidOverlay(Ellipse)
% 在三视图（XZ、XY、YZ）上叠加椭球截面
% Ellipse: a, b, c, centerx, centery, centerz
% x_range, y_range, z_range：对应 imagesc 的坐标向量

    t = linspace(0,2*pi,300);

    % XZ 截面
    x_xz = Ellipse.centerx + Ellipse.a * cos(t);
    z_xz = Ellipse.centerz + Ellipse.c * sin(t);

    % YZ 截面
    y_yz = Ellipse.centery + Ellipse.b * cos(t);
    z_yz = Ellipse.centerz + Ellipse.c * sin(t);

    % XY 截面
    x_xy = Ellipse.centerx + Ellipse.a * cos(t);
    y_xy = Ellipse.centery + Ellipse.b * sin(t);

    % XZ 投影（subplot 1）
    subplot(1,3,1);
    hold on;
    % patch(z_xz, x_xz, [0 0.4470 0.7410], 'FaceAlpha', 0.1, ...
    %     'EdgeColor',[0.4940 0.1840 0.5560],'LineStyle',"--",'LineWidth',2);
    plot(z_xz, x_xz, 'r-', 'LineWidth', 1.5);
    hold off;

    % XY 投影（subplot 2）
    subplot(1,3,2);
    hold on;
    % patch(x_xy, y_xy, [0 0.4470 0.7410], 'FaceAlpha', 0.1, ...
    %     'EdgeColor',[0.4940 0.1840 0.5560],'LineStyle',"--",'LineWidth',2);
    plot(x_xy, y_xy, 'r-', 'LineWidth', 1.5);
    hold off;

    % YZ 投影（subplot 3）
    subplot(1,3,3);
    hold on;
    % patch(z_yz, y_yz, [0 0.4470 0.7410], 'FaceAlpha', 0.1, ...
    %     'EdgeColor',[0.4940 0.1840 0.5560],'LineStyle',"--",'LineWidth',2);
    plot(z_yz, y_yz, 'r-', 'LineWidth', 1.5);
    hold off;
end
