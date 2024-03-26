% Implementation of the perceptual model proposed in S. van de Par, A.
% Kohlrausch, R. Heusdens, J. Jensen, S. H. Jensen, "A Perceptual Model for
% Sinusoidal Audio Coding Based on Spectral Integration", EURASIP Journal
% on Applied Signal Processing, 2005:9, pp. 1292-1304.

classdef perceptualModel < handle
    properties
        % correspondance between digital RMS and pressure RMS in Pascal
        fullscalePressureInPa = 1; % [Pa]
        
        % Inverse hearing threshold
        outerMiddleEarMagnitudeResponse = [];

        % Gammatone filterbank response
        filterbankMagnitudeResponse = [];
        numberOfChannels = 0;

        % Combined weighting
        channelFrequencyMagnitudeResponse = [];

        blockSize = 0;
        % Normalization constants
        Ca = 0;
        Cs = 0;
        Leff = 1;

        squaredWeightingCurve = [];
    end
    methods 
        function perceptualModel = perceptualModel(blockSize, Fs, fullscalePressureInDbSpl)
            if ~(mod(blockSize,2) == 0)
                error('Block size is expected to be even');
            end
            perceptualModel.blockSize = blockSize;
            fullscalePressureInPa = 10^(fullscalePressureInDbSpl/20) * 20e-6;
            perceptualModel.fullscalePressureInPa = fullscalePressureInPa;
            frequency = (0:Fs/blockSize:Fs/2).';

            % Determine outer middle ear response
            thresholdOfHearingInDbSpl = interpolatedThresholdOfHearing(frequency);
%             thresholdOfHearingInDbSpl = interpolatedThresholdOfHearing(frequency,'painter_2000');
%             thresholdOfHearingInDbSpl = interpolatedThresholdOfHearing(frequency,'none');
            thresholdOfHearingInPa = 10.^(thresholdOfHearingInDbSpl/20)*20e-6;
            % determine the threshold of hearing in the digital scale where
            % 0 dBFS = fullscalePressureInDbSpl
            thresholdOfHearing = thresholdOfHearingInPa * 1/fullscalePressureInPa;
            perceptualModel.outerMiddleEarMagnitudeResponse = 1./thresholdOfHearing;

            % Determine Gammatone filterbank
            perceptualModel.filterbankMagnitudeResponse = gammatoneFilterResponse(0, Fs/2, frequency);
            perceptualModel.numberOfChannels = size(perceptualModel.filterbankMagnitudeResponse,2);
            perceptualModel.channelFrequencyMagnitudeResponse = ...
                repmat(perceptualModel.outerMiddleEarMagnitudeResponse,1,perceptualModel.numberOfChannels) ...
                .* perceptualModel.filterbankMagnitudeResponse;

            % Determine normalization constants
            perceptualModel.Leff = min(blockSize/Fs * 1/0.3, 1); 
            
            % Calibrate the model as suggested in the original paper
            % Determine amplitude of 52 dB SPL sinusoid (for sinusoids the
            % amplitude is sqrt(2) x the RMS value)
            A52 = sqrt(2) * 10^(52/20)*20e-6 * 1/fullscalePressureInPa;
            % Determine amplitude of 70 dB SPL sinusoid (for sinusoids the
            % amplitude is sqrt(2) x the RMS value)
            A70 = sqrt(2) * 10^(70/20)*20e-6 * 1/fullscalePressureInPa;
            % Choose a frequency for the calibration
            fIdx = floor(blockSize/48);
            calibrationFrequency = frequency(fIdx);
            time = (0:1/Fs:blockSize/Fs-1/Fs)';
            sine52 = A52*sin(2*pi*calibrationFrequency*time);
            sine70 = A70*sin(2*pi*calibrationFrequency*time);
            spectrum52 = sqrt(2)/blockSize*fft(sine52,blockSize);
            spectrum70 = sqrt(2)/blockSize*fft(sine70,blockSize);

            S52 = abs(spectrum52(fIdx));
            S70 = abs(spectrum70(fIdx));

            K = sum(perceptualModel.filterbankMagnitudeResponse(fIdx,:).^2) * perceptualModel.Leff;
            k52 = perceptualModel.channelFrequencyMagnitudeResponse(fIdx,:).^2 * S52^2;
            k70 = perceptualModel.channelFrequencyMagnitudeResponse(fIdx,:).^2 * S70^2;
            fun = @(x) (perceptualModel.Leff*sum(k52./(k70 + x*K)) - 1./x);
            xNeg = 1e-1;
            xPos = 200;
            if fun(xPos) < 0
                xPos = 1000;
            end
            if sign(fun(xNeg)) == sign(fun(xPos))
                keyboard
                error('Initialization of bisection method failed');
            end
            tolerance = 1e-6;
            maxIterations = 1000;
            itr = 1;
            solutionFound = false;
            while itr < maxIterations && ~solutionFound
                xMid = (xPos + xNeg)/2;
                fMid = fun(xMid);
                if fMid == 0 || (xPos - xNeg)/2 < tolerance
                    solutionFound = true;
                end
                itr = itr + 1;
                if sign(fMid) == sign(fun(xNeg))
                    xNeg = xMid;
                else
                    xPos = xMid;
                end
            end
            if itr >= maxIterations
                error('Bisection method reached max iterations. Solution failed');
            end
            if abs(fun(xMid)) > 1e-3
                keyboard;
            end
            perceptualModel.Cs = xMid;
            perceptualModel.Ca = xMid*K;
        end

        function obj = determineSquaredWeightingCurve(obj, inputBlock)
            arguments (Input)
                obj (1,1) perceptualModel
                inputBlock (:,1) double
            end
            arguments (Output)
                obj (1,1) perceptualModel
            end
            if length(inputBlock) ~= obj.blockSize
                error('The size of the inputBlock does not match the expected blockSize');
            end
            if any(~isreal(inputBlock))
                blockSpectrum = inputBlock;
            else
                blockSpectrum = sqrt(2)/obj.blockSize*fft(inputBlock);
            end
            spectrum = repmat(abs(blockSpectrum(1:obj.blockSize/2+1)), 1, obj.numberOfChannels);
            maskerPower = sum( (obj.channelFrequencyMagnitudeResponse.*spectrum).^2, 1);
            obj.squaredWeightingCurve = obj.Cs*obj.Leff* sum( obj.channelFrequencyMagnitudeResponse.^2 ...
                ./( repmat(maskerPower,obj.blockSize/2+1,1) + obj.Ca ), 2 );
        end

        function squaredMaskingCurve = getSquaredMaskingCurve(obj)
            arguments (Input)
                obj (1,1) perceptualModel
            end
            arguments (Output)
                squaredMaskingCurve (:,1) double
            end
            squaredMaskingCurve = 1./obj.squaredWeightingCurve;
        end

        function weightingCurve = getWeightingCurve(obj)
            arguments (Input)
                obj (1,1) perceptualModel
            end
            arguments (Output)
                weightingCurve (:,1) double
            end
            tmpCurve = [obj.squaredWeightingCurve; flipud(obj.squaredWeightingCurve(2:end-1))];
            weightingCurve = sqrt(tmpCurve);
        end

        function weightingCurve = getNormalizedWeightingCurve(obj)
            % Function returning the weighting curve normalized such that
            % it is centered around 1, rather than 20e-6 Pa.
            arguments (Input)
                obj (1,1) perceptualModel
            end
            arguments (Output)
                weightingCurve (:,1) double
            end
            tmpCurve = [obj.squaredWeightingCurve; flipud(obj.squaredWeightingCurve(2:end-1))];
            tmpCurve = sqrt(tmpCurve);
            tmpCurve = tmpCurve * 20e-6;
            weightingCurve = tmpCurve;
        end

        function weightingCurve = getUnitVectorWeightingCurve(obj)
            % Function returning the weighting curve normalized such that
            % it has unit length. 
            arguments (Input)
                obj (1,1) perceptualModel
            end
            arguments (Output)
                weightingCurve (:,1) double
            end
            tmpCurve = [obj.squaredWeightingCurve; flipud(obj.squaredWeightingCurve(2:end-1))];
            tmpCurve = sqrt(tmpCurve);
            tmpCurve = tmpCurve/norm(tmpCurve);
            weightingCurve = tmpCurve;
        end
    end
end