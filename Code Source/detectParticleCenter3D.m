function [idx_center, coord_center, peak_val] = detectParticleCenter3D(vol, x_range, y_range, z_range)
vol = double(vol);

[peak_val, linear_idx] = max(vol(:));
[iy, ix, iz] = ind2sub(size(vol), linear_idx);
idx_center = [iy, ix, iz];

x_range = x_range(:);
y_range = y_range(:);
z_range = z_range(:);

coord_center = [x_range(ix), y_range(iy), z_range(iz)];
end
