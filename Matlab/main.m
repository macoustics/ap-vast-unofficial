% Example file show-casing how to use the MATLAB-implementation of apVast.
% Note that the model operates with a calibrated threshold of hearing. This
% means that the input audiosignals convolved with the room impulse
% responses should correspond to a sound pressure. In the given example,
% the correspondance is set to 94 dB SPL for a 0 dB digital signal, i.e., a
% pure-tone of RMS-value 1 digital corresponds to a pure-tone of RMS-value
% 1 Pascal in sound pressure (94 dB SPL re. 20e-6 Pa). If this scaling is
% chosen wrongly, the threshold of hearing might completely mask the audio.

%% Initialize
close all
clear all
clc

% Set paths to source files
foldername = fullfile('./');
path = genpath(foldername);
addpath(path)

% Initialize basic parameters written in the init function
FsCalc = 8000;

[signalA, orgFs] = audioread('music_01_norm.wav');
signalA = resample(2*10*signalA(:,1), FsCalc, orgFs);
[signalB, orgFs] = audioread('music_02_norm.wav');
signalB = resample(10*signalB(:,1), FsCalc, orgFs);

weight = 'apvast';
Comment = '';


%% Calculate source weights
switch weight
    case 'apvast'
        load('rirs.mat')
        FIRLength = 400;
        blockSize = 1020;
        numberOfEigenvectors = [1; FIRLength*10/2; FIRLength*10];
        ModellingDelay = 50;
        RefIdxA = 1;
        RefIdxB = 1;
        mu = 1;
        statBufferLength = 1020;
        apvastObj = apVast(blockSize, b_control_rir, d_control_rir, FIRLength, ModellingDelay, RefIdxA, RefIdxB, numberOfEigenvectors, mu, statBufferLength, FsCalc, 94, true);
        hopSize = blockSize/2;
        numberOfHops = 20;
        numberOfSolutions = length(numberOfEigenvectors);
        outputA = zeros(hopSize*numberOfHops, 10, numberOfSolutions);
        outputB = zeros(hopSize*numberOfHops, 10, numberOfSolutions);
        targetA = zeros(hopSize*numberOfHops, 10);
        targetB = zeros(hopSize*numberOfHops, 10);
        for hIdx = 0 : numberOfHops-1
            disp(['hIdx = ' int2str(hIdx) ' / ' int2str(numberOfHops-1)]);
            idx = hIdx*hopSize + (1:hopSize);
            tic
            [tmpA, tmpB, targA, targB] = apvastObj.processInputBuffer(signalA(idx),signalB(idx));
            toc
            outputA(idx,:, :) = tmpA;
            outputB(idx,:, :) = tmpB;
            targetA(idx,:) = targA;
            targetB(idx,:) = targB;
        end
        
        numberOfEvalMics = size(b_validation_rir,3);
        pressureAtoA = zeros(hopSize*numberOfHops, numberOfEvalMics, numberOfSolutions);
        pressureAtoB = pressureAtoA;
        pressureBtoA = pressureAtoA;
        pressureBtoB = pressureAtoA;
        for sIdx = 1:numberOfSolutions
            pressureAtoA(:,:,sIdx) = predictPressure(outputA(:,:,sIdx), b_validation_rir);
            pressureAtoB(:,:,sIdx) = predictPressure(outputA(:,:,sIdx), d_validation_rir);
            pressureBtoA(:,:,sIdx) = predictPressure(outputB(:,:,sIdx), b_validation_rir);
            pressureBtoB(:,:,sIdx) = predictPressure(outputB(:,:,sIdx), d_validation_rir);
        end
        targetPressureA = predictPressure(targetA, b_validation_rir);
        targetPressureB = predictPressure(targetB, d_validation_rir);
        
        lim=[-1, 1]*.5;
        figure
        subplot(2,2,1)
        plot(targetPressureA(:,1));
        hold on; grid on;
        plot(pressureAtoA(:,1,1));
        plot(pressureAtoA(:,1,2));
        plot(pressureAtoA(:,1,3));
        title('A to A')
        legend('target','V = 1','V = JL/2','V = JL')
        ylim(lim)

        subplot(2,2,2)
        plot(targetPressureB(:,1));
        hold on; grid on;
        plot(pressureBtoB(:,1,1));
        plot(pressureBtoB(:,1,2));
        plot(pressureBtoB(:,1,3));
        title('B to B')
        legend('target','V = 1','V = JL/2','V = JL')
        ylim(lim)

        subplot(2,2,4)
        plot(targetPressureB(:,1));
        hold on; grid on;
        plot(pressureAtoB(:,1,1));
        plot(pressureAtoB(:,1,2));
        plot(pressureAtoB(:,1,3));
        title('A to B')
        legend('target','V = 1','V = JL/2','V = JL')
        ylim(lim)

        subplot(2,2,3)
        plot(targetPressureA(:,1));
        hold on; grid on;
        plot(pressureBtoA(:,1,1));
        plot(pressureBtoA(:,1,2));
        plot(pressureBtoA(:,1,3));
        title('B to A')
        legend('target','V = 1','V = JL/2','V = JL')
        ylim(lim)

        nmseA = 0;
        nmseB = 0;
        for mIdx = 1:numberOfEvalMics
            nmseA = nmseA + norm(targetPressureA(:,mIdx) - pressureAtoA(:,mIdx,3))^2/norm(targetPressureA(:,mIdx))^2;
            nmseB = nmseB + norm(targetPressureB(:,mIdx) - pressureBtoB(:,mIdx,3))^2/norm(targetPressureB(:,mIdx))^2;
        end
        nmseA = nmseA/numberOfEvalMics;
        nmseB = nmseB/numberOfEvalMics;

        acA = 10*log10(norm(pressureAtoA(:,:,3),'fro')^2/norm(pressureAtoB(:,:,3),'fro')^2);
        acB = 10*log10(norm(pressureBtoB(:,:,3),'fro')^2/norm(pressureBtoA(:,:,3),'fro')^2);
        
    otherwise
        error([weight ' is not recognized as an available control algorithm'])
end


