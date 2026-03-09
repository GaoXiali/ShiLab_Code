function saveCase13SpotStats(spotStats, outputDir)
if nargin < 2 || isempty(outputDir)
    outputDir = 'PAresult';
end

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

summaryTable = struct2table(spotStats.summary);
profileTable = struct2table(spotStats.profiles);

writetable(summaryTable, fullfile(outputDir, 'spot_summary_case13.csv'));
writetable(profileTable, fullfile(outputDir, 'spot_profiles_case13.csv'));
save(fullfile(outputDir, 'spot_profiles_case13.mat'), 'summaryTable', 'profileTable');
end
