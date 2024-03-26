% Function for validating the calibration of the perceptual model. This
% should correspond to Fig. 2b in S. van de Par, A.
% Kohlrausch, R. Heusdens, J. Jensen, S. H. Jensen, "A Perceptual Model for
% Sinusoidal Audio Coding Based on Spectral Integration", EURASIP Journal
% on Applied Signal Processing, 2005:9, pp. 1292-1304.
%
% Note that when the Gammatone filters are spaced by 1 ERB the response
% will exhibit ripples.

clear all; close all; clc

Fs = 48000;
blockSize = 4800;
time = (0:1/Fs:blockSize/Fs-1/Fs)';
A50 = sqrt(2) * 10^(50/20)*20e-6;
signalSine50 = A50 * sin(2*pi*1000*time);
signalZero = zeros(blockSize,1);

pModel = perceptualModel(blockSize, Fs, 94);

pModel.determineSquaredWeightingCurve(signalZero);
threshold = pModel.getSquaredMaskingCurve;

pModel.determineSquaredWeightingCurve(signalSine50);
maskingCurve = pModel.getSquaredMaskingCurve;

frequency = (0:Fs/blockSize:Fs/2)';
figure
semilogx(frequency, 10*log10(threshold/(20e-6)^2));
hold on; grid on;
semilogx(frequency, 10*log10(maskingCurve/(20e-6)^2));
frequency = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500]';
refThreshold = interpolatedThresholdOfHearing(frequency);
semilogx(frequency, refThreshold,'o');
xlabel('Frequency [Hz]')
ylabel('SPL [dB]')
legend('Threshold in quiet','Masking curve','Threshold in quiet from ISO226:2003')
xlim([20 15e3]);
ylim([-10 100])
