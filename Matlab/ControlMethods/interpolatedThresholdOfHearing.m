function [thresholdInDb] = interpolatedThresholdOfHearing(frequency, varargin)
% Function for calculating the threshold of hearing in dB SPL at the
% specified frequency vector.

if isempty(varargin)
    method = 'iso226_2003';
else 
    method = lower(varargin{1});
end

if strcmp(method,'none')
    thresholdInDb = zeros(size(frequency));
elseif strcmp(method, 'painter_2000')
    % Approximate expression (eq. 1) used in T. Painter, and A. Spanias,
    % "Perceptual Conding of Digital Audio", Proceedings of the IEEE, Vol.
    % 88, No. 4, 2000, pp. 451-513.
    thresholdInDb = 3.64*(frequency/1000).^(-0.8) - 6.5*exp(-0.6*(frequency/1000 - 3.3).^2) + 10^(-3)*(frequency/1000).^4;
else
    [refThreshold, refFrequency] = getThreshold(method);
    thresholdInDb = interp1(refFrequency, refThreshold, frequency,"spline");
end

end

function [spl, frequency] = getThreshold(method)

switch method
    case 'iso226_2003'
        frequency = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500]';
        spl = [78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5, 1.7, -1.3, -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3]';
    otherwise
        error('No valid threshold method provided. Consider the default iso226_2003')
end
end