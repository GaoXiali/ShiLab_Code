%% **************************斐波那契阵列展示*******************************
%
% 本代码用于半球阵光声成像系统中，按照斐波那契阵列排布的换能器位置简单可视化
%
% Author：Xiali Gao  Version：1.0  2025.4.10
%**************************************************************************
close all
addpath 'D:\GXL\DataRead';
dector=load('coordinate.txt'); % 加载探测器坐标文件

%绘制探测器点
x = dector(:,1);
y = dector(:,2);
z = -dector(:,3);

figure(1);
plot3(x,y,z,'b.');axis equal;hold on
zlim([-120,20])

for i = 1:20
    % 坐标旋转(角度制,顺时针为正，逆时针为负）
    theta_x = 0; % 以x轴为中心旋转 
    theta_y = 0; % 以y轴为中心旋转 
    theta_z = 0.8*i; % 以z轴为中心旋转 
    
    %坐标平移(这里的坐标平移会影响椭球中心选取)
    trans_x = 0;
    trans_y = 0;
    trans_z = 0;
    
    rotate_x_mat = [1 0 0 0;0 cosd(theta_x) -sind(theta_x),0;0 sind(theta_x) cosd(theta_x) 0;0 0 0 1];
    rotate_y_mat = [cosd(theta_y) 0 -sind(theta_y) 0;0 1 0,0;sind(theta_y) 0 cosd(theta_y) 0;0 0 0 1];
    rotate_z_mat = [cosd(theta_z) -sind(theta_z) 0 0;sind(theta_z) cosd(theta_z) 0,0;0 0 1 0;0 0 0 1];
    trans_mat = [1 0 0 trans_x;0 1 0 trans_y;0 0 1 trans_z;0 0 0 1];
    afine_mat = trans_mat*rotate_x_mat*rotate_y_mat*rotate_z_mat; %以原点为中心，先转z轴，再转y轴，再转x轴，最后平移
    
    %绘制旋转后的探测器点
    dector_new=[dector,dector(:,1)*0+1]*afine_mat'; 
    x_new = dector_new(:,1);
    y_new = dector_new(:,2);
    z_new = -dector_new(:,3);
    
    plot3(x_new,y_new,z_new,'b.');axis equal;hold on
    zlim([-120,20])
    pause(0.2)
end

% for i = 1:1:1024
%     figure(2)
%     plot3(x_new(i),y_new(i),z_new(i),'.');axis equal;hold on
%     xlim([-120,120]);ylim([-120,120]);zlim([-120,20]);view(2)
%     % pause()
% end