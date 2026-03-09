function [fwhm, peakVal, halfVal, leftCross, rightCross] = computeProfileFWHM(coord, profile)
coord = coord(:);
profile = profile(:);

fwhm = NaN;
peakVal = NaN;
halfVal = NaN;
leftCross = NaN;
rightCross = NaN;

validMask = isfinite(coord) & isfinite(profile);
coord = coord(validMask);
profile = profile(validMask);
if numel(coord) < 3
    return;
end

[peakVal, peakIdx] = max(profile);
halfVal = peakVal / 2;

leftIdx = find(profile(1:peakIdx) < halfVal, 1, 'last');
if isempty(leftIdx) || leftIdx >= peakIdx
    return;
end

rightRelIdx = find(profile(peakIdx:end) < halfVal, 1, 'first');
if isempty(rightRelIdx)
    return;
end
rightIdx = peakIdx + rightRelIdx - 1;
if rightIdx <= peakIdx
    return;
end

leftCross = linearCross(coord(leftIdx), coord(leftIdx + 1), profile(leftIdx), profile(leftIdx + 1), halfVal);
rightCross = linearCross(coord(rightIdx - 1), coord(rightIdx), profile(rightIdx - 1), profile(rightIdx), halfVal);

if isfinite(leftCross) && isfinite(rightCross)
    fwhm = rightCross - leftCross;
end
end

function x = linearCross(x1, x2, y1, y2, yTarget)
if y2 == y1
    x = (x1 + x2) / 2;
    return;
end
x = x1 + (yTarget - y1) * (x2 - x1) / (y2 - y1);
end
