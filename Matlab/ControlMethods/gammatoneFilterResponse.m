% Subfunction for generating the amplitude response of ERB-spaced Gammatone
% filters. Amplitude response expression from S. van de Par, A.
% Kohlrausch, R. Heusdens, J. Jensen, S. H. Jensen, "A Perceptual Model for
% Sinusoidal Audio Coding Based on Spectral Integration", EURASIP Journal
% on Applied Signal Processing, 2005:9, pp. 1292-1304.

function magnitudeResponse = gammatoneFilterResponse(flow, fhigh, frequency)
% function for designing the amplitude response of a Gammatone filterbank
% where the Gammatone filters are spaced uniformly on the ERB scale and
% have 1 ERB bandwidth.
filterOrder = 4;

[centerFrequencies, bandwidths] = calculateCenterFrequencies(flow, fhigh);
k = 2^(filterOrder-1) * factorial(filterOrder-1) / (pi * doubleFactorial(2*filterOrder-3));
frequency = repmat(frequency(:), 1, length(centerFrequencies));
centerFrequencies = repmat(centerFrequencies.',size(frequency,1),1);
bandwidths = repmat(bandwidths.', size(frequency,1), 1);

magnitudeResponse = (1 + ((frequency - centerFrequencies)./(k*bandwidths)).^2 ).^(-filterOrder/2);

% figure
% semilogx(frequency, 20*log10(magnitudeResponse));
% hold on; grid on;
% semilogx(frequency, 10*log10(sum(magnitudeResponse.^2,2)));




end

%% Helper functions
function [centerFrequencies, bandwidths] = calculateCenterFrequencies(flow, fhigh)
bandwidth = 1; % 1 ERB bandwidth
% Convert frequency [Hz] limits to erbs
% erbLimits = 21.4*log10(4.37*[flow; fhigh]/1000 + 1);
% erbLimits = 2302.6/(24.673*4.368)*log10(1+[flow; fhigh]*0.004368);
erbLimits = 9.2645*sign([flow; fhigh]) .* log(1 + [flow; fhigh]*0.00437);
erbRange = erbLimits(2) - erbLimits(1);

% Calculate number of points excluding final point (bandwidth = 1 erb)
n = floor (erbRange/bandwidth);

% The remainder is calculated in order to center the points correctly
% between fmin and fmax
remainder = erbRange - n*bandwidth;
erbPoints = erbLimits(1) + (0:n)'*bandwidth+remainder/2;

% Convert erb center points to frequency [Hz]
centerFrequencies = (1/0.00437)*sign(erbPoints).*(exp(abs(erbPoints)/9.2645) - 1);
bandwidths = 24.7 + centerFrequencies./9.265;
% bandwidths = 24.7*(4.37*centerFrequencies/1000 + 1);
end

function result = doubleFactorial(number)
    if number == 0
        result = 1;
    elseif mod(number,2) == 0
        result = prod(2:2:number);
    else
        result = prod(1:2:number);
    end
end

