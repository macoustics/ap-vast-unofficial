% Implementation of the block based sound zones algorithm described in T.
% Lee, J. K. Nielsen, M. G. Christensen, "Signal-Adaptive and Perceptually
% Optimized Sound Zones With Variable Spand Trade-Off Filters", IEEE/ACM
% Transactions on Audio, Speech, and Language Processing, vol. 28, 2020,
% pp. 2412-2426.

classdef apVast < handle
    properties
        % WOLA processing
        m_hopSize = 0;
        m_blockSize = 0;
        m_window = [];
        m_inputABlock = [];
        m_inputBBlock = [];

        % RIR + statistics
        m_rirLength = 0;
        m_numberOfMics = 0;
        m_numberOfSrcs = 0;
        m_rirA = [];
        m_rirB = [];
        m_targetRirA = [];
        m_targetRirB = [];
        m_rirAtoAState = [];
        m_rirAtoBState = [];
        m_targetRirAtoAState = [];
        m_rirBtoAState = [];
        m_rirBtoBState = [];
        m_targetRirBtoBState = [];
        

        % Control filters
        m_filterLength;
        m_modellingDelay = 0;
        m_referenceIndexA = 0;
        m_referenceIndexB = 0;
        m_filters = [];
        m_mu = 0;
        m_numberOfEigenvectors = 0;
        m_filterSpectraA = [];
        m_filterSpectraB = [];
        m_calculateZoneB = true;

        % Loudspeaker response buffers
        m_loudspeakerResponseAtoABuffer = [];
        m_loudspeakerResponseAtoBBuffer = [];
        m_loudspeakerResponseBtoABuffer = [];
        m_loudspeakerResponseBtoBBuffer = [];
        m_loudspeakerTargetResponseAtoABuffer = [];
        m_loudspeakerTargetResponseBtoBBuffer = [];

        % Perceptual model
        m_Fs = 0;
        m_perceptualModel = [];
        m_forwardScale = 0; % Used to compute spectrum in dB SPL
        m_inverseScale = 0; % Used to compute back to digital representation

        % Weighted loudspeaker response overlap buffers
        m_loudspeakerWeightedResponseAtoAOverlapBuffer = [];
        m_loudspeakerWeightedResponseAtoBOverlapBuffer = [];
        m_loudspeakerWeightedResponseBtoAOverlapBuffer = [];
        m_loudspeakerWeightedResponseBtoBOverlapBuffer = [];
        m_loudspeakerWeightedTargetResponseAtoAOverlapBuffer = [];
        m_loudspeakerWeightedTargetResponseBtoBOverlapBuffer = [];

        % Weighted loudspeaker response buffers (used for computing the
        % statistics)
        m_statisticsBufferLength = 0;
        m_loudspeakerWeightedResponseAtoABuffer = [];
        m_loudspeakerWeightedResponseAtoBBuffer = [];
        m_loudspeakerWeightedResponseBtoABuffer = [];
        m_loudspeakerWeightedResponseBtoBBuffer = [];
        m_loudspeakerWeightedTargetResponseAtoABuffer = [];
        m_loudspeakerWeightedTargetResponseBtoBBuffer = [];
        m_RAtoA = [];
        m_RAtoB = [];
        m_RBtoA = [];
        m_RBtoB = [];
        m_rA =  [];
        m_rB =  [];

        % Perceptual weighting
        m_weightingSpectraA = [];
        m_weightingSpectraB = [];

        % Output overlap buffers
        m_numberOfSolutions = 0;
        m_outputAOverlapBuffer = [];
        m_outputBOverlapBuffer = [];
        m_outputTargetAOverlapBuffer = [];
        m_outputTargetBOverlapBuffer = [];
    end
    methods 
        function apVast = apVast(blockSize, rirA, rirB, filterLength, ModellingDelay, ReferenceIndexA, ReferenceIndexB, numberOfEigenvectors, mu, statisticsBufferLength, Fs, pressureScale, calcZoneB)
            % Input parameters:
            % -----------------------
            % blockSize     Integer
            %               Size of the input buffer blocks that will be
            %               processed.
            % rirB:         Matrix (ndim = 3)
            %               Impulse responses from each source to each microphone in 
            %               the bright zone. Size = [rirLength, numberOfSrc, numberOfMics];
            % rirD:         Matrix (ndim = 3)
            %               Impulse responses from each source to each microphone in
            %               the dark zone. Size = [rirLength, numberOfSrc, numberOfMics];
            % filterLength  Scalar
            %               Length of FIR filter for each loudspeaker
            % ModellingDelay    Integer
            %               Introduced modelling delay in samples.
            % ReferenceIndex:   Integer in [1,numberOfSrcs]
            %               Index for the reference loudspeaker used to define the
            %               target audio signal in the bright zone.
            % numberOfEigenvectors   1D Integer array
            %               Number of eigenvectors from the joint diagonalization,
            %               which are used to approximate the solution. If mu = 1,
            %               numberOfEigenvectors = 1 corresponds to the BACC solution
            %               and numberOfEigenvectors = filterLength*numberOfSrcs corresponds to
            %               pressure matching. If
            %               length(numberOfEigenvectors) > 1, one solution
            %               per element will be returned.
            % mu            Scalar \in [0,1]
            %               Adjustment parameter for the variable span linear filters
            % statisticsBufferLength    Integer
            %               Size (in samples) of temporary buffer used to calculate the
            %               covariance matrices
            % Fs            Integer
            %               Sampling rate
            % pressureScale Scalar
            %               pressure in dB SPL corresponding to 0 dBFS in
            %               the digital representation
            % calcZoneB     boolean
            %               true if filters should be determined for zoneB
            %               otherwise false. If false the filters will stay
            %               at zero.

            % Initialize WOLA parameters
            apVast.m_blockSize = blockSize;
            apVast.m_hopSize = blockSize/2;
            if mod(blockSize,2) ~= 0
                error('BlockSize must be an even number');
            end
            apVast.m_window =  sin(pi/blockSize*(0:(blockSize-1)).');
            apVast.m_inputABlock = zeros(blockSize,1);
            apVast.m_inputBBlock = zeros(blockSize,1);

            % Initialize RIR parameters
            apVast.m_rirA = rirA;
            apVast.m_rirB = rirB;
            apVast.m_rirLength = size(rirB,1);
            apVast.m_numberOfSrcs = size(rirB,2);
            apVast.m_numberOfMics = size(rirB,3);
            apVast.m_targetRirA = zeros(apVast.m_rirLength, apVast.m_numberOfMics);
            apVast.m_targetRirB = zeros(apVast.m_rirLength, apVast.m_numberOfMics);
            apVast.m_modellingDelay = ModellingDelay;
            apVast.m_referenceIndexA = ReferenceIndexA;
            apVast.m_referenceIndexB = ReferenceIndexB;
            for mIdx = 1:apVast.m_numberOfMics
                apVast.m_targetRirA(:,mIdx) = [zeros(ModellingDelay,1); squeeze(rirA(1:apVast.m_rirLength-ModellingDelay,ReferenceIndexA,mIdx))];
                apVast.m_targetRirB(:,mIdx) = [zeros(ModellingDelay,1); squeeze(rirB(1:apVast.m_rirLength-ModellingDelay,ReferenceIndexB,mIdx))];
            end
            apVast.m_rirAtoAState = zeros(apVast.m_rirLength-1,apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_rirAtoBState = zeros(apVast.m_rirLength-1,apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_rirBtoAState = zeros(apVast.m_rirLength-1,apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_rirBtoBState = zeros(apVast.m_rirLength-1,apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_targetRirAtoAState = zeros(apVast.m_rirLength-1, apVast.m_numberOfMics);
            apVast.m_targetRirBtoBState = zeros(apVast.m_rirLength-1, apVast.m_numberOfMics);
            
            % Initialize control filters
            apVast.m_filterLength = filterLength;
            apVast.m_filters = zeros(filterLength, apVast.m_numberOfSrcs);
            apVast.m_mu = mu;
            apVast.m_numberOfEigenvectors = numberOfEigenvectors;

            % Initialize loudspeaker response buffers
            apVast.m_loudspeakerResponseAtoABuffer = zeros(blockSize, apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_loudspeakerResponseAtoBBuffer = zeros(blockSize, apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_loudspeakerResponseBtoABuffer = zeros(blockSize, apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_loudspeakerResponseBtoBBuffer = zeros(blockSize, apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_loudspeakerTargetResponseAtoABuffer = zeros(blockSize, apVast.m_numberOfMics);
            apVast.m_loudspeakerTargetResponseBtoBBuffer = zeros(blockSize, apVast.m_numberOfMics);

            % Initialize loudspeaker weighted response overlap buffers
            apVast.m_loudspeakerWeightedResponseAtoAOverlapBuffer = zeros(blockSize, apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_loudspeakerWeightedResponseAtoBOverlapBuffer = zeros(blockSize, apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_loudspeakerWeightedResponseBtoAOverlapBuffer = zeros(blockSize, apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_loudspeakerWeightedResponseBtoBOverlapBuffer = zeros(blockSize, apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_loudspeakerWeightedTargetResponseAtoAOverlapBuffer = zeros(blockSize, apVast.m_numberOfMics);
            apVast.m_loudspeakerWeightedTargetResponseBtoBOverlapBuffer = zeros(blockSize, apVast.m_numberOfMics);

            % Initialize loudspeaker weighted response buffers (used for
            % computing the statistics)
            apVast.m_statisticsBufferLength = statisticsBufferLength;
            if statisticsBufferLength < 2*filterLength
                error('The statisticsBufferLength is chosen smaller than the 2*filterLength-1. This ensures that the sample covariance matrices are guaranteed to be rank deficient.')
            end
            apVast.m_loudspeakerWeightedResponseAtoABuffer = zeros(statisticsBufferLength, apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_loudspeakerWeightedResponseAtoBBuffer = zeros(statisticsBufferLength, apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_loudspeakerWeightedResponseBtoABuffer = zeros(statisticsBufferLength, apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_loudspeakerWeightedResponseBtoBBuffer = zeros(statisticsBufferLength, apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_loudspeakerWeightedTargetResponseAtoABuffer = zeros(statisticsBufferLength, apVast.m_numberOfMics);
            apVast.m_loudspeakerWeightedTargetResponseBtoBBuffer = zeros(statisticsBufferLength, apVast.m_numberOfMics);

            % Initialize output overlap buffers
            apVast.m_numberOfSolutions = length(numberOfEigenvectors);
            apVast.m_outputAOverlapBuffer = zeros(blockSize, apVast.m_numberOfSrcs, apVast.m_numberOfSolutions);
            apVast.m_outputBOverlapBuffer = zeros(blockSize, apVast.m_numberOfSrcs, apVast.m_numberOfSolutions);
            apVast.m_outputTargetAOverlapBuffer = zeros(blockSize, apVast.m_numberOfSrcs);
            apVast.m_outputTargetBOverlapBuffer = zeros(blockSize, apVast.m_numberOfSrcs);

            % Initialize perceptual model
            apVast.m_Fs = Fs;
            apVast.m_perceptualModel = perceptualModel(blockSize, Fs, pressureScale);
            apVast.m_forwardScale = sqrt(2)/blockSize;
            apVast.m_inverseScale = blockSize/sqrt(2);

            % Switch for calculating zone B
            apVast.m_calculateZoneB = calcZoneB;
        end

        %% Main function (public interface)
        function [outputBuffersA, outputBuffersB, targetOutputBuffersA, targetOutputBuffersB, obj] = processInputBuffer(obj, inputA, inputB)
            arguments (Input)
                obj (1,1) apVast
                inputA (:, 1)
                inputB (:, 1)
            end
            arguments (Output)
                outputBuffersA (:,:,:) double
                outputBuffersB (:,:,:) double
                targetOutputBuffersA (:,:) double
                targetOutputBuffersB (:,:) double
                obj (1,1) apVast
            end
            if length(inputA) ~= obj.m_hopSize
                error(['The inputA buffer should be of length obj.hopSize = ' int2str(obj.m_hopSize)]);
            end
            obj.updateLoudspeakerResponseBuffers(inputA, inputB);
            obj.updateWeightedTargetSignals();
            obj.updateWeightedLoudspeakerResponses();
            obj.updateStatistics();
            obj.calculateFilterSpectra(obj.m_mu, obj.m_numberOfEigenvectors);
            obj.updateInputBlocks(inputA, inputB);
            [outputBuffersA, outputBuffersB, obj] = obj.computeOutputBuffers();
            [targetOutputBuffersA, targetOutputBuffersB, obj] = obj.computeTargetOutputBuffers();
        end

        %% Helper functions (should be private)
        function [obj] = updateLoudspeakerResponseBuffers(obj, inputA, inputB)
            arguments (Input)
                obj (1,1) apVast
                inputA (:, 1)
                inputB (:, 1)
            end
            arguments (Output)
                obj (1,1) apVast
            end
            idx = (obj.m_hopSize + 1 : obj.m_blockSize);
            for mIdx = 1:obj.m_numberOfMics
                % Update targetA to zone A
                [tmpInput, tmpState] = filter(obj.m_targetRirA(:,mIdx),1, inputA, obj.m_targetRirAtoAState(:,mIdx));
                obj.m_targetRirAtoAState(:,mIdx) = tmpState;
                obj.m_loudspeakerTargetResponseAtoABuffer(:,mIdx) = [obj.m_loudspeakerTargetResponseAtoABuffer(idx,mIdx); tmpInput];
                % Update targetB to zone B
                [tmpInput, tmpState] = filter(obj.m_targetRirB(:,mIdx),1, inputB, obj.m_targetRirBtoBState(:,mIdx));
                obj.m_targetRirBtoBState(:,mIdx) = tmpState;
                obj.m_loudspeakerTargetResponseBtoBBuffer(:,mIdx) = [obj.m_loudspeakerTargetResponseBtoBBuffer(idx,mIdx); tmpInput];
                for sIdx = 1:obj.m_numberOfSrcs
                    % Update inputA to zone A
                    [tmpInput, tmpState] = filter(obj.m_rirA(:,sIdx,mIdx),1, inputA, obj.m_rirAtoAState(:,sIdx,mIdx));
                    obj.m_rirAtoAState(:,sIdx,mIdx) = tmpState;
                    obj.m_loudspeakerResponseAtoABuffer(:,sIdx,mIdx) = [obj.m_loudspeakerResponseAtoABuffer(idx,sIdx,mIdx); tmpInput];
                    % Update inputA to zone B
                    [tmpInput, tmpState] = filter(obj.m_rirB(:,sIdx,mIdx),1, inputA, obj.m_rirAtoBState(:,sIdx,mIdx));
                    obj.m_rirAtoBState(:,sIdx,mIdx) = tmpState;
                    obj.m_loudspeakerResponseAtoBBuffer(:,sIdx,mIdx) = [obj.m_loudspeakerResponseAtoBBuffer(idx,sIdx,mIdx); tmpInput];
                    % Update inputB to zone A
                    [tmpInput, tmpState] = filter(obj.m_rirA(:,sIdx,mIdx),1, inputB, obj.m_rirBtoAState(:,sIdx,mIdx));
                    obj.m_rirBtoAState(:,sIdx,mIdx) = tmpState;
                    obj.m_loudspeakerResponseBtoABuffer(:,sIdx,mIdx) = [obj.m_loudspeakerResponseBtoABuffer(idx,sIdx,mIdx); tmpInput];
                    % Update inputB to zone B
                    [tmpInput, tmpState] = filter(obj.m_rirB(:,sIdx,mIdx),1, inputB, obj.m_rirBtoBState(:,sIdx,mIdx));
                    obj.m_rirBtoBState(:,sIdx,mIdx) = tmpState;
                    obj.m_loudspeakerResponseBtoBBuffer(:,sIdx,mIdx) = [obj.m_loudspeakerResponseBtoBBuffer(idx,sIdx,mIdx); tmpInput];
                end
            end
        end

        function [obj] = updateWeightedTargetSignals(obj)
            arguments (Input)
                obj (1,1) apVast
            end
            arguments (Output)
                obj (1,1) apVast
            end
            % Calculate spectra
            targetAtoASpectra = zeros(obj.m_blockSize, obj.m_numberOfMics);
            targetBtoBSpectra = zeros(obj.m_blockSize, obj.m_numberOfMics);
            for mIdx = 1 : obj.m_numberOfMics
                targetAtoASpectra(:,mIdx) = obj.m_forwardScale*fft(obj.m_window .* obj.m_loudspeakerTargetResponseAtoABuffer(:,mIdx));
                targetBtoBSpectra(:,mIdx) = obj.m_forwardScale*fft(obj.m_window .* obj.m_loudspeakerTargetResponseBtoBBuffer(:,mIdx));
            end

            obj.updatePerceptualWeighting(targetAtoASpectra, targetBtoBSpectra);

            % Circular convolution with weighting filter
            targetAtoASpectra = targetAtoASpectra .* obj.m_weightingSpectraA;
            targetBtoBSpectra = targetBtoBSpectra .* obj.m_weightingSpectraB;

            % WOLA reconstruction
            for mIdx = 1 : obj.m_numberOfMics
                % Zone A
                tmpOld = obj.m_loudspeakerWeightedTargetResponseAtoAOverlapBuffer(:,mIdx);
                tmpNew = obj.m_inverseScale * obj.m_window .* ifft(targetAtoASpectra(:,mIdx));
                obj.m_loudspeakerWeightedTargetResponseAtoAOverlapBuffer(:,mIdx) = [tmpOld(obj.m_hopSize+1:obj.m_blockSize); zeros(obj.m_hopSize,1)] + tmpNew;
                % Zone B
                tmpOld = obj.m_loudspeakerWeightedTargetResponseBtoBOverlapBuffer(:,mIdx);
                tmpNew = obj.m_inverseScale * obj.m_window .* ifft(targetBtoBSpectra(:,mIdx));
                obj.m_loudspeakerWeightedTargetResponseBtoBOverlapBuffer(:,mIdx) = [tmpOld(obj.m_hopSize+1:obj.m_blockSize); zeros(obj.m_hopSize,1)] + tmpNew;
            end

            % Update weightedTargetResponseBuffers
            idx = (obj.m_hopSize + 1 : obj.m_statisticsBufferLength);
            for mIdx = 1 : obj.m_numberOfMics
                obj.m_loudspeakerWeightedTargetResponseAtoABuffer(:,mIdx) = [obj.m_loudspeakerWeightedTargetResponseAtoABuffer(idx,mIdx); obj.m_loudspeakerWeightedTargetResponseAtoAOverlapBuffer(1:obj.m_hopSize,mIdx)];
                obj.m_loudspeakerWeightedTargetResponseBtoBBuffer(:,mIdx) = [obj.m_loudspeakerWeightedTargetResponseBtoBBuffer(idx,mIdx); obj.m_loudspeakerWeightedTargetResponseBtoBOverlapBuffer(1:obj.m_hopSize,mIdx)];
            end
        end

        function [obj] = updateWeightedLoudspeakerResponses(obj)
            arguments (Input)
                obj (1,1) apVast
            end
            arguments (Output)
                obj (1,1) apVast
            end
            % Calculate spectra
            AtoASpectra = zeros(obj.m_blockSize, obj.m_numberOfSrcs, obj.m_numberOfMics);
            AtoBSpectra = zeros(obj.m_blockSize, obj.m_numberOfSrcs, obj.m_numberOfMics);
            BtoASpectra = zeros(obj.m_blockSize, obj.m_numberOfSrcs, obj.m_numberOfMics);
            BtoBSpectra = zeros(obj.m_blockSize, obj.m_numberOfSrcs, obj.m_numberOfMics);
            for mIdx = 1 : obj.m_numberOfMics
                AtoASpectra(:,:,mIdx) = obj.m_forwardScale * fft(repmat(obj.m_window, 1, obj.m_numberOfSrcs) .* obj.m_loudspeakerResponseAtoABuffer(:,:,mIdx));
                AtoBSpectra(:,:,mIdx) = obj.m_forwardScale * fft(repmat(obj.m_window, 1, obj.m_numberOfSrcs) .* obj.m_loudspeakerResponseAtoBBuffer(:,:,mIdx));
                BtoASpectra(:,:,mIdx) = obj.m_forwardScale * fft(repmat(obj.m_window, 1, obj.m_numberOfSrcs) .* obj.m_loudspeakerResponseBtoABuffer(:,:,mIdx));
                BtoBSpectra(:,:,mIdx) = obj.m_forwardScale * fft(repmat(obj.m_window, 1, obj.m_numberOfSrcs) .* obj.m_loudspeakerResponseBtoBBuffer(:,:,mIdx));
            end

            % Circular convolution with weighting filter
            for mIdx = 1 : obj.m_numberOfMics
                AtoASpectra(:,:,mIdx) = AtoASpectra(:,:,mIdx) .* repmat(obj.m_weightingSpectraA(:,mIdx),1,obj.m_numberOfSrcs);
                AtoBSpectra(:,:,mIdx) = AtoBSpectra(:,:,mIdx) .* repmat(obj.m_weightingSpectraB(:,mIdx),1,obj.m_numberOfSrcs);
                BtoASpectra(:,:,mIdx) = BtoASpectra(:,:,mIdx) .* repmat(obj.m_weightingSpectraA(:,mIdx),1,obj.m_numberOfSrcs);
                BtoBSpectra(:,:,mIdx) = BtoBSpectra(:,:,mIdx) .* repmat(obj.m_weightingSpectraB(:,mIdx),1,obj.m_numberOfSrcs);
            end

            % WOLA reconstruction
            for mIdx = 1 : obj.m_numberOfMics
                % Signal A to Zone A
                tmpOld = obj.m_loudspeakerWeightedResponseAtoAOverlapBuffer(:,:,mIdx);
                tmpNew = obj.m_inverseScale * repmat(obj.m_window,1,obj.m_numberOfSrcs) .* ifft(AtoASpectra(:,:,mIdx), obj.m_blockSize, 1);
                obj.m_loudspeakerWeightedResponseAtoAOverlapBuffer(:,:,mIdx) = [tmpOld(obj.m_hopSize+1:obj.m_blockSize,:); zeros(obj.m_hopSize,obj.m_numberOfSrcs)] + tmpNew;
                % Signal A to Zone B
                tmpOld = obj.m_loudspeakerWeightedResponseAtoBOverlapBuffer(:,:,mIdx);
                tmpNew = obj.m_inverseScale * repmat(obj.m_window,1,obj.m_numberOfSrcs) .* ifft(AtoBSpectra(:,:,mIdx), obj.m_blockSize, 1);
                obj.m_loudspeakerWeightedResponseAtoBOverlapBuffer(:,:,mIdx) = [tmpOld(obj.m_hopSize+1:obj.m_blockSize,:); zeros(obj.m_hopSize,obj.m_numberOfSrcs)] + tmpNew;
                % Signal B to Zone A
                tmpOld = obj.m_loudspeakerWeightedResponseBtoAOverlapBuffer(:,:,mIdx);
                tmpNew = obj.m_inverseScale * repmat(obj.m_window,1,obj.m_numberOfSrcs) .* ifft(BtoASpectra(:,:,mIdx), obj.m_blockSize, 1);
                obj.m_loudspeakerWeightedResponseBtoAOverlapBuffer(:,:,mIdx) = [tmpOld(obj.m_hopSize+1:obj.m_blockSize,:); zeros(obj.m_hopSize,obj.m_numberOfSrcs)] + tmpNew;
                % Signal B to Zone B
                tmpOld = obj.m_loudspeakerWeightedResponseBtoBOverlapBuffer(:,:,mIdx);
                tmpNew = obj.m_inverseScale * repmat(obj.m_window,1,obj.m_numberOfSrcs) .* ifft(BtoBSpectra(:,:,mIdx), obj.m_blockSize, 1);
                obj.m_loudspeakerWeightedResponseBtoBOverlapBuffer(:,:,mIdx) = [tmpOld(obj.m_hopSize+1:obj.m_blockSize,:); zeros(obj.m_hopSize,obj.m_numberOfSrcs)] + tmpNew;
            end

            % Update weightedTargetResponseBuffers
            idx = (obj.m_hopSize + 1 : obj.m_statisticsBufferLength);
            for mIdx = 1 : obj.m_numberOfMics
                obj.m_loudspeakerWeightedResponseAtoABuffer(:,:,mIdx) = [obj.m_loudspeakerWeightedResponseAtoABuffer(idx,:,mIdx); obj.m_loudspeakerWeightedResponseAtoAOverlapBuffer(1:obj.m_hopSize,:,mIdx)];
                obj.m_loudspeakerWeightedResponseAtoBBuffer(:,:,mIdx) = [obj.m_loudspeakerWeightedResponseAtoBBuffer(idx,:,mIdx); obj.m_loudspeakerWeightedResponseAtoBOverlapBuffer(1:obj.m_hopSize,:,mIdx)];
                obj.m_loudspeakerWeightedResponseBtoABuffer(:,:,mIdx) = [obj.m_loudspeakerWeightedResponseBtoABuffer(idx,:,mIdx); obj.m_loudspeakerWeightedResponseBtoAOverlapBuffer(1:obj.m_hopSize,:,mIdx)];
                obj.m_loudspeakerWeightedResponseBtoBBuffer(:,:,mIdx) = [obj.m_loudspeakerWeightedResponseBtoBBuffer(idx,:,mIdx); obj.m_loudspeakerWeightedResponseBtoBOverlapBuffer(1:obj.m_hopSize,:,mIdx)];
            end
        end

        function [obj] = updatePerceptualWeighting(obj,targetAtoASpectra, targetBtoBSpectra)
            arguments (Input)
                obj (1,1) apVast
                targetAtoASpectra (:,:) double
                targetBtoBSpectra (:,:) double
            end
            arguments (Output)
                obj (1,1) apVast
            end
            % Update perceptual weighting (currently we have no weighting)
            obj.m_weightingSpectraA = ones(obj.m_blockSize, obj.m_numberOfMics);
            obj.m_weightingSpectraB = ones(obj.m_blockSize, obj.m_numberOfMics);
            for mIdx = 1 : obj.m_numberOfMics
                obj.m_perceptualModel.determineSquaredWeightingCurve(targetAtoASpectra(:,mIdx));
                obj.m_weightingSpectraA(:,mIdx) = obj.m_perceptualModel.getUnitVectorWeightingCurve;
%                 obj.m_weightingSpectraA(:,mIdx) = obj.m_perceptualModel.getNormalizedWeightingCurve;

                obj.m_perceptualModel.determineSquaredWeightingCurve(targetBtoBSpectra(:,mIdx));
                obj.m_weightingSpectraB(:,mIdx) = obj.m_perceptualModel.getUnitVectorWeightingCurve;
%                 obj.m_weightingSpectraB(:,mIdx) = obj.m_perceptualModel.getNormalizedWeightingCurve;
            end
            
        end

        function [obj] = updateStatistics(obj)
            arguments (Input)
                obj (1,1) apVast
            end
            arguments (Output)
                obj (1,1) apVast
            end
            obj.resetStatistics();

            for mIdx = 1 : obj.m_numberOfMics
                Y = zeros(obj.m_filterLength*obj.m_numberOfSrcs, obj.m_statisticsBufferLength - obj.m_filterLength + 1);
                for sIdx = 0 : obj.m_numberOfSrcs - 1
                    Y(sIdx*obj.m_filterLength + (1:obj.m_filterLength),:) = toeplitz(flipud(obj.m_loudspeakerWeightedResponseAtoABuffer(1:obj.m_filterLength,sIdx+1,mIdx)), obj.m_loudspeakerWeightedResponseAtoABuffer(obj.m_filterLength:end,sIdx+1,mIdx).');
                end
                obj.m_RAtoA = obj.m_RAtoA + Y*Y.';
                obj.m_rA = obj.m_rA + Y * (obj.m_loudspeakerWeightedTargetResponseAtoABuffer(obj.m_filterLength:end,mIdx));

                Y = zeros(obj.m_filterLength*obj.m_numberOfSrcs, obj.m_statisticsBufferLength - obj.m_filterLength + 1);
                for sIdx = 0 : obj.m_numberOfSrcs - 1
                    Y(sIdx*obj.m_filterLength + (1:obj.m_filterLength),:) = toeplitz(flipud(obj.m_loudspeakerWeightedResponseAtoBBuffer(1:obj.m_filterLength,sIdx+1,mIdx)), obj.m_loudspeakerWeightedResponseAtoBBuffer(obj.m_filterLength:end,sIdx+1,mIdx).');
                end
                obj.m_RAtoB = obj.m_RAtoB + Y*Y.';

                if obj.m_calculateZoneB
                    Y = zeros(obj.m_filterLength*obj.m_numberOfSrcs, obj.m_statisticsBufferLength - obj.m_filterLength + 1);
                    for sIdx = 0 : obj.m_numberOfSrcs - 1
                        Y(sIdx*obj.m_filterLength + (1:obj.m_filterLength),:) = toeplitz(flipud(obj.m_loudspeakerWeightedResponseBtoABuffer(1:obj.m_filterLength,sIdx+1,mIdx)), obj.m_loudspeakerWeightedResponseBtoABuffer(obj.m_filterLength:end,sIdx+1,mIdx).');
                    end
                    obj.m_RBtoA = obj.m_RBtoA + Y*Y.';
    
                    Y = zeros(obj.m_filterLength*obj.m_numberOfSrcs, obj.m_statisticsBufferLength - obj.m_filterLength + 1);
                    for sIdx = 0 : obj.m_numberOfSrcs - 1
                        Y(sIdx*obj.m_filterLength + (1:obj.m_filterLength),:) = toeplitz(flipud(obj.m_loudspeakerWeightedResponseBtoBBuffer(1:obj.m_filterLength,sIdx+1,mIdx)), obj.m_loudspeakerWeightedResponseBtoBBuffer(obj.m_filterLength:end,sIdx+1,mIdx).');
                    end
                    obj.m_RBtoB = obj.m_RBtoB + Y*Y.';
                    obj.m_rB = obj.m_rB + Y * (obj.m_loudspeakerWeightedTargetResponseBtoBBuffer(obj.m_filterLength:end,mIdx));
                end
            end
            normalizationFactor = ((obj.m_statisticsBufferLength-obj.m_filterLength+1)*obj.m_numberOfMics);
            obj.m_RAtoA = obj.m_RAtoA/normalizationFactor;
            obj.m_rA = obj.m_rA/normalizationFactor;
            obj.m_RAtoB = obj.m_RAtoB/normalizationFactor;
            if obj.m_calculateZoneB
                obj.m_RBtoA = obj.m_RBtoA/normalizationFactor;
                obj.m_RBtoB = obj.m_RBtoB/normalizationFactor;
                obj.m_rB = obj.m_rB/normalizationFactor;
            end

            % Depreciated implementation (SLOW):
%             for nIdx = 1 : obj.m_statisticsBufferLength - obj.m_filterLength + 1
%                 idx = nIdx + (0:obj.m_filterLength-1);
%                 for mIdx = 1 : obj.m_numberOfMics
%                     tmp = flipud(obj.m_loudspeakerWeightedResponseAtoABuffer(idx,:,mIdx));
%                     yAtoA = tmp(:);
%                     tmp = flipud(obj.m_loudspeakerWeightedResponseAtoBBuffer(idx,:,mIdx));
%                     yAtoB = tmp(:);
%                     tmp = flipud(obj.m_loudspeakerWeightedResponseBtoABuffer(idx,:,mIdx));
%                     yBtoA = tmp(:);
%                     tmp = flipud(obj.m_loudspeakerWeightedResponseBtoBBuffer(idx,:,mIdx));
%                     yBtoB = tmp(:);
% 
%                     dA = flipud(obj.m_loudspeakerWeightedTargetResponseAtoABuffer(idx,mIdx));
%                     dB = flipud(obj.m_loudspeakerWeightedTargetResponseBtoBBuffer(idx,mIdx));
% 
%                     obj.m_RAtoA = obj.m_RAtoA + yAtoA * yAtoA';
%                     obj.m_RAtoB = obj.m_RAtoB + yAtoB * yAtoB';
%                     obj.m_RBtoA = obj.m_RBtoA + yBtoA * yBtoA';
%                     obj.m_RBtoB = obj.m_RBtoB + yBtoB * yBtoB';
% 
%                     obj.m_rA = obj.m_rA + yAtoA * dA(1);
%                     obj.m_rB = obj.m_rB + yBtoB * dB(1);
%                 end
%             end

        end

        function [obj] = resetStatistics(obj)
            arguments (Input)
                obj (1,1) apVast
            end
            arguments (Output)
                obj (1,1) apVast
            end
            obj.m_RAtoA = zeros(obj.m_filterLength*obj.m_numberOfSrcs);
            obj.m_RAtoB = zeros(obj.m_filterLength*obj.m_numberOfSrcs);
            obj.m_RBtoA = zeros(obj.m_filterLength*obj.m_numberOfSrcs);
            obj.m_RBtoB = zeros(obj.m_filterLength*obj.m_numberOfSrcs);
            obj.m_rA = zeros(obj.m_filterLength*obj.m_numberOfSrcs,1);
            obj.m_rB = zeros(obj.m_filterLength*obj.m_numberOfSrcs,1);
        end

        function [obj] = calculateFilterSpectra(obj, mu, numberOfEigenvectors)
            arguments (Input)
                obj (1,1) apVast
                mu (1,1) double
                numberOfEigenvectors (:,1) double
            end
            arguments (Output)
                obj (1,1) apVast
            end
            % Joint diagonalization
            obj.diagonalLoading();

            try
                [UA,lambdaA] = jdiag(obj.m_RAtoA, obj.m_RAtoB, 'vector', false);
            catch
                keyboard
            end
            if obj.m_calculateZoneB
                [UB,lambdaB] = jdiag(obj.m_RBtoB, obj.m_RBtoA, 'vector', false);
            end
            
            % Determine filters
            wA = zeros(obj.m_filterLength*obj.m_numberOfSrcs,obj.m_numberOfSolutions);
            wB = zeros(obj.m_filterLength*obj.m_numberOfSrcs,obj.m_numberOfSolutions);
            obj.m_filterSpectraA = zeros(obj.m_blockSize, obj.m_numberOfSrcs, obj.m_numberOfSolutions);
            obj.m_filterSpectraB = zeros(obj.m_blockSize, obj.m_numberOfSrcs, obj.m_numberOfSolutions);
            for sIdx = 1:obj.m_numberOfSolutions
                if sIdx == 1
                    for i = 1:numberOfEigenvectors(sIdx)
                        wA(:,sIdx) = wA(:,sIdx) + (UA(:,i).'*obj.m_rA)/(lambdaA(i) + mu) * UA(:,i);
                        if obj.m_calculateZoneB
                            wB(:,sIdx) = wB(:,sIdx) + (UB(:,i).'*obj.m_rB)/(lambdaB(i) + mu) * UB(:,i);
                        end
                    end
                else
                    wA(:,sIdx) = wA(:,sIdx-1);
                    wB(:,sIdx) = wB(:,sIdx-1);
                    for i = numberOfEigenvectors(sIdx-1) + 1:numberOfEigenvectors(sIdx)
                        wA(:,sIdx) = wA(:,sIdx) + (UA(:,i).'*obj.m_rA)/(lambdaA(i) + mu) * UA(:,i);
                        if obj.m_calculateZoneB
                            wB(:,sIdx) = wB(:,sIdx) + (UB(:,i).'*obj.m_rB)/(lambdaB(i) + mu) * UB(:,i);
                        end
                    end
                end
    
                % Determine filter spectra
                obj.m_filterSpectraA(:,:,sIdx) = fft(reshape(wA(:,sIdx), obj.m_filterLength, obj.m_numberOfSrcs), obj.m_blockSize, 1);
                obj.m_filterSpectraB(:,:,sIdx) = fft(reshape(wB(:,sIdx), obj.m_filterLength, obj.m_numberOfSrcs), obj.m_blockSize, 1);
            end
        end

        function [obj] = diagonalLoading(obj)
            arguments (Input)
                obj (1,1) apVast
            end
            arguments (Output)
                obj (1,1) apVast
            end
            darkCondLimit = 5e-3;
            brightCondLimit = 1e-8;

            obj.m_RAtoA = obj.m_RAtoA + brightCondLimit*eye(size(obj.m_RAtoA)) * norm(obj.m_RAtoA);
            obj.m_RAtoB = obj.m_RAtoB + darkCondLimit*eye(size(obj.m_RAtoB)) * norm(obj.m_RAtoB);
            if obj.m_calculateZoneB
                obj.m_RBtoA = obj.m_RBtoA + darkCondLimit*eye(size(obj.m_RBtoA)) * norm(obj.m_RBtoA);
                obj.m_RBtoB = obj.m_RBtoB + brightCondLimit*eye(size(obj.m_RBtoB)) * norm(obj.m_RBtoB);
            end

        end

        function [obj] = updateInputBlocks(obj, inputA, inputB)
            arguments (Input)
                obj (1,1) apVast
                inputA (:, 1)
                inputB (:, 1)
            end
            arguments (Output)
                obj (1,1) apVast
            end
            obj.m_inputABlock = [obj.m_inputABlock(obj.m_hopSize + 1 : obj.m_blockSize); inputA];
            obj.m_inputBBlock = [obj.m_inputBBlock(obj.m_hopSize + 1 : obj.m_blockSize); inputB];
        end

        function [outputBuffersA, outputBuffersB, obj] = computeTargetOutputBuffers(obj)
            arguments (Input)
                obj (1,1) apVast
            end
            arguments (Output)
                outputBuffersA (:,:) double
                outputBuffersB (:,:) double
                obj (1,1) apVast
            end
            % Compute input spectra
            inputSpectrumA = fft(obj.m_window .* obj.m_inputABlock);
            inputSpectrumB = fft(obj.m_window .* obj.m_inputBBlock);

            targetFiltersA = zeros(obj.m_filterLength, obj.m_numberOfSrcs);
            targetFiltersB = zeros(obj.m_filterLength, obj.m_numberOfSrcs);
            targetFiltersA(obj.m_modellingDelay+1, obj.m_referenceIndexA) = 1;
            targetFiltersB(obj.m_modellingDelay+1, obj.m_referenceIndexB) = 1;
            targetFilterSpectraA = fft(targetFiltersA, obj.m_blockSize, 1);
            targetFilterSpectraB = fft(targetFiltersB, obj.m_blockSize, 1);

            % Circular convolution with the filter spectra
            outputSpectraA = repmat(inputSpectrumA, 1, obj.m_numberOfSrcs) .* targetFilterSpectraA;
            outputSpectraB = repmat(inputSpectrumB, 1, obj.m_numberOfSrcs) .* targetFilterSpectraB;

            % Update the output overlap buffers
            idx = obj.m_hopSize + 1:obj.m_blockSize;
            obj.m_outputTargetAOverlapBuffer = [obj.m_outputTargetAOverlapBuffer(idx,:); zeros(obj.m_hopSize,obj.m_numberOfSrcs)] + ifft(outputSpectraA, obj.m_blockSize, 1) .* repmat(obj.m_window, 1, obj.m_numberOfSrcs);
            obj.m_outputTargetBOverlapBuffer = [obj.m_outputTargetBOverlapBuffer(idx,:); zeros(obj.m_hopSize,obj.m_numberOfSrcs)] + ifft(outputSpectraB, obj.m_blockSize, 1) .* repmat(obj.m_window, 1, obj.m_numberOfSrcs);

            % Extract samples for the output buffers
            outputBuffersA = obj.m_outputTargetAOverlapBuffer(1:obj.m_hopSize,:);
            outputBuffersB = obj.m_outputTargetBOverlapBuffer(1:obj.m_hopSize,:);
        end

        function [outputBuffersA, outputBuffersB, obj] = computeOutputBuffers(obj)
            arguments (Input)
                obj (1,1) apVast
            end
            arguments (Output)
                outputBuffersA (:,:,:) double
                outputBuffersB (:,:,:) double
                obj (1,1) apVast
            end
            % Compute input spectra
            inputSpectrumA = fft(obj.m_window .* obj.m_inputABlock);
            inputSpectrumB = fft(obj.m_window .* obj.m_inputBBlock);
            
            outputBuffersA = zeros(obj.m_hopSize, obj.m_numberOfSrcs, obj.m_numberOfSolutions);
            outputBuffersB = zeros(obj.m_hopSize, obj.m_numberOfSrcs, obj.m_numberOfSolutions);
            for sIdx = 1:obj.m_numberOfSolutions
                % Circular convolution with the filter spectra
                outputSpectraA = repmat(inputSpectrumA, 1, obj.m_numberOfSrcs) .* obj.m_filterSpectraA(:,:,sIdx);
                outputSpectraB = repmat(inputSpectrumB, 1, obj.m_numberOfSrcs) .* obj.m_filterSpectraB(:,:,sIdx);
    
                % Update the output overlap buffers
                idx = obj.m_hopSize + 1:obj.m_blockSize;
                obj.m_outputAOverlapBuffer(:,:,sIdx) = [obj.m_outputAOverlapBuffer(idx,:,sIdx); zeros(obj.m_hopSize,obj.m_numberOfSrcs)] + ifft(outputSpectraA, obj.m_blockSize, 1) .* repmat(obj.m_window, 1, obj.m_numberOfSrcs);
                obj.m_outputBOverlapBuffer(:,:,sIdx) = [obj.m_outputBOverlapBuffer(idx,:,sIdx); zeros(obj.m_hopSize,obj.m_numberOfSrcs)] + ifft(outputSpectraB, obj.m_blockSize, 1) .* repmat(obj.m_window, 1, obj.m_numberOfSrcs);
    
                % Extract samples for the output buffers
                outputBuffersA(:,:,sIdx) = obj.m_outputAOverlapBuffer(1:obj.m_hopSize,:,sIdx);
                outputBuffersB(:,:,sIdx) = obj.m_outputBOverlapBuffer(1:obj.m_hopSize,:,sIdx);
            end
        end

        
  
    end
end
