%% Initialize
close all
clear all
clc

tic

% Set paths to source files
foldername = fullfile('./');
path = genpath(foldername);
addpath(path)

% Initialize basic parameters written in the init function
FsCalc = 1200;

% ===========================
%       Switches
% ---------------------------
[signalA, orgFs] = audioread('Daft Punk - Doin It Right.wav');
signalA = resample(signalA, FsCalc, orgFs);
[signalB, orgFs] = audioread('ACDC - Back in black.wav');
signalB = resample(signalB, FsCalc, orgFs);

room = 'Measurement'; % ['Measurement']
lenIR = 0;
WindowOn = false; % [true, false] windowing of bright / dark zones
task = 'GenerateFilters'; % ['GenerateFilters','EvalPerformance']

weight = 'vast'; % ['BACC_galvez','vast']
RegParam = 0*1e-3;
FIRLength = 100;
Comment = '';
ZoneInversion = true; % [true, false] Switch for choosing which zone is bright and which is dark

%% Transferfunctions
switch room
    case 'Measurement'
        load Measurements/VR-lab/ZoneAWooferImpulseResponsesResampled.mat
        % Permute from [samples, mics, drivers] to [mics, samples, drivers]
        IRa1 = permute(Imp,[2,1,3]);
        load Measurements/VR-lab/ZoneAWooferImpulseResponsesResampledRepeat.mat
        % Permute from [samples, mics, drivers] to [mics, samples, drivers]
        IRa2 = permute(Imp,[2,1,3]);
        load Measurements/VR-lab/ZoneBWooferImpulseResponsesResampled.mat
        % Permute from [samples, mics, drivers] to [mics, samples, drivers]
        IRb1 = permute(Imp,[2,1,3]);
        load Measurements/VR-lab/ZoneBWooferImpulseResponsesResampledRepeat.mat
        % Permute from [samples, mics, drivers] to [mics, samples, drivers]
        IRb2 = permute(Imp,[2,1,3]);
end

Mics = 1:9;
Taps = 1:320;       
IRStart = 1;

Drivers = [1:8];
if ~ZoneInversion
    IR_Target = IRa1(Mics,Taps,Drivers);
    IR_Dark = IRb1(Mics,Taps,Drivers);  
    IR_TargetEval = IRa2(:,Taps,Drivers);
    IR_DarkEval = IRb2(:,Taps,Drivers);
else
    IR_Target = IRb1(Mics,Taps,Drivers);
    IR_Dark = IRa1(Mics,Taps,Drivers);
    IR_TargetEval = IRb2(:,Taps,Drivers);
    IR_DarkEval = IRa2(:,Taps,Drivers);
end

switch task
    case 'GenerateFilters'
        %% Calculate source weights
        switch weight
            case 'vast'
                numberOfEigenvectors = FIRLength/2;
                ModellingDelay = 20;
                RefIdx = 5;
                mu = 1;
                [h] = vast(IR_Target,IR_Dark,FIRLength, ModellingDelay, RefIdx, numberOfEigenvectors, mu);
                h = h.';
                EvalSolutionPaper(IR_Target, IR_Dark, h./norm(h,'fro'), FsCalc);
            case 'apvast'
                blockSize = 2*length(Taps);
                numberOfEigenvectors = FIRLength/2;
                ModellingDelay = 20;
                RefIdx = 5;
                mu = 1;
                statBufferLength = 1000;
                apvastObj = apVast(blockSize, permute(IR_Target, [2,3,1]), permute(IR_Dark, [2,3,1]),FIRLength, ModellingDelay, RefIdx, RefIdx, numberOfEigenvectors, mu, statBufferLength);
                hopSize = blockSize/2;
                numberOfHops = 10;
                outputA = zeros(hopSize*numberOfHops, length(Drivers));
                outputB = zeros(hopSize*numberOfHops, length(Drivers));
                for hIdx = 0 : numberOfHops-1
                    disp(['hIdx = ' int2str(hIdx) ' / ' int2str(numberOfHops-1)]);
                    idx = hIdx*hopSize + (1:hopSize);
                    [tmpA, tmpB] = apvastObj.processInputBuffer(signalA(idx),signalB(idx));
                    outputA(idx,:) = tmpA;
                    outputB(idx,:) = tmpB;
                end

                pressureAtoA = predictPressure(outputA, permute(IR_Target, [2,3,1]));
                pressureAtoB = predictPressure(outputA, permute(IR_Dark, [2,3,1]));
                pressureBtoA = predictPressure(outputB, permute(IR_Target, [2,3,1]));
                pressureBtoB = predictPressure(outputB, permute(IR_Dark, [2,3,1]));

                figure
                subplot(2,2,1)
                plot(pressureAtoA);
                title('A to A')

                subplot(2,2,2)
                plot(pressureBtoB);
                title('B to B')

                subplot(2,2,3)
                plot(pressureAtoB);
                title('A to B')

                subplot(2,2,4)
                plot(pressureBtoA);
                title('B to A')
                

                
            otherwise
                error([weight ' is not recognized as an available control algorithm'])
        end
    otherwise
        error([task ' is not recognized as an available task'])
end
toc

