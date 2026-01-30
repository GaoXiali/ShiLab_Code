function fixMatlabFilenames(folder_path)
    % 获取目录下所有 .mat 文件
    % 注意：这里建议获取所有文件，以便进行 4 个一组的逻辑判断
    file_struct = dir(fullfile(folder_path, 'Data_*.mat'));
    
    % 1. 确保文件按名称排序（dir的结果在某些系统中不一定有序）
    [~, sortIdx] = sort({file_struct.name});
    file_struct = file_struct(sortIdx);
    
    num_files = length(file_struct);
    fprintf('找到 %d 个文件，准备开始检查...\n', num_files);
    
    % 2. 每 4 个一组进行处理
    for i = 1:4:num_files
        % 检查最后一组是否完整
        if i + 3 > num_files
            warning('剩余文件不足 4 个，起始索引为 %d，已跳过。', i);
            break;
        end
        
        % 获取当前组的“模板”文件名（即该组第 1 个文件，索引 0）
        % 假设格式为：Data_20260114_192341_0_0.mat
        template_name = file_struct(i).name;
        
        % 使用下划线分割文件名
        % parts{1}='Data', {2}='日期', {3}='时间戳', {4}='组内编号', {5}='0.mat'
        parts = strsplit(template_name, '_');
        
        if length(parts) < 4
            fprintf('跳过格式不符的文件: %s\n', template_name);
            continue;
        end
        
        % 提取基准前缀（Data_YYYYMMDD_HHMMSS）
        base_prefix = strjoin(parts(1:3), '_');
        suffix_fixed = parts{5}; % 获取最后的 '0.mat'
        
        % 3. 检查并修正该组内的后续 3 个文件 (j=1, 2, 3)
        for j = 1:3
            current_idx = i + j;
            actual_name = file_struct(current_idx).name;
            
            % 构造期望的文件名：基准前缀 + _j_ + 固定结尾
            expected_name = sprintf('%s_%d_%s', base_prefix, j, suffix_fixed);
            
            % 如果实际名称与期望名称不符，则重命名
            if ~strcmp(actual_name, expected_name)
                old_path = fullfile(folder_path, actual_name);
                new_path = fullfile(folder_path, expected_name);
                
                fprintf('发现错误: [%s] -> 修正为 [%s]\n', actual_name, expected_name);
                
                try
                    movefile(old_path, new_path);
                catch ME
                    fprintf('重命名失败: %s\n', ME.message);
                end
            end
        end
    end
    fprintf('检查完成。\n');
end