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
        m_filters = [];
        m_mu = 0;
        m_numberOfEigenvectors = 0;
        m_filterSpectraA = [];
        m_filterSpectraB = [];

        % Loudspeaker response buffers
        m_loudspeakerResponseAtoABuffer = [];
        m_loudspeakerResponseAtoBBuffer = [];
        m_loudspeakerResponseBtoABuffer = [];
        m_loudspeakerResponseBtoBBuffer = [];
        m_loudspeakerTargetResponseAtoABuffer = [];
        m_loudspeakerTargetResponseBtoBBuffer = [];

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

        % Intermediate results (used for debugging)
        m_UA = [];
        m_UB = [];
        m_wA = [];
        m_wB = [];
        m_lambdaA = [];
        m_lambdaB = [];

        % Perceptual weighting
        m_weightingSpectraA = [];
        m_weightingSpectraB = [];

        % Output overlap buffers
        m_outputAOverlapBuffer = [];
        m_outputBOverlapBuffer = [];
    end
    methods 
        function apVast = apVast(blockSize, rirA, rirB, filterLength, ModellingDelay, ReferenceIndexA, ReferenceIndexB, numberOfEigenvectors, mu, statisticsBufferLength)
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
            % numberOfEigenvectors   Integer
            %               Number of eigenvectors from the joint diagonalization,
            %               which are used to approximate the solution. If mu = 1,
            %               numberOfEigenvectors = 1 corresponds to the BACC solution
            %               and numberOfEigenvectors = filterLength*numberOfSrcs corresponds to
            %               pressure matching.
            % mu            Scalar \in [0,1]
            %               Adjustment parameter for the variable span linear filters

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
            for mIdx = 1:apVast.m_numberOfMics
                apVast.m_targetRirA(:,mIdx) = [zeros(ModellingDelay,1); squeeze(rirA(1:apVast.m_rirLength-ModellingDelay,ReferenceIndexA,mIdx))];
                apVast.m_targetRirB(:,mIdx) = [zeros(ModellingDelay,1); squeeze(rirB(1:apVast.m_rirLength-ModellingDelay,ReferenceIndexB,mIdx))];
            end
            apVast.m_rirAtoAState = zeros(apVast.m_rirLength-1,apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_rirAtoBState = zeros(apVast.m_rirLength-1,apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_targetRirAtoAState = zeros(apVast.m_rirLength-1, apVast.m_numberOfMics);
            apVast.m_rirBtoAState = zeros(apVast.m_rirLength-1,apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_rirBtoBState = zeros(apVast.m_rirLength-1,apVast.m_numberOfSrcs, apVast.m_numberOfMics);
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
            apVast.m_loudspeakerWeightedResponseAtoABuffer = zeros(statisticsBufferLength, apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_loudspeakerWeightedResponseAtoBBuffer = zeros(statisticsBufferLength, apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_loudspeakerWeightedResponseBtoABuffer = zeros(statisticsBufferLength, apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_loudspeakerWeightedResponseBtoBBuffer = zeros(statisticsBufferLength, apVast.m_numberOfSrcs, apVast.m_numberOfMics);
            apVast.m_loudspeakerWeightedTargetResponseAtoABuffer = zeros(statisticsBufferLength, apVast.m_numberOfMics);
            apVast.m_loudspeakerWeightedTargetResponseBtoBBuffer = zeros(statisticsBufferLength, apVast.m_numberOfMics);

            % Initialize output overlap buffers
            apVast.m_outputAOverlapBuffer = zeros(blockSize, apVast.m_numberOfSrcs);
            apVast.m_outputBOverlapBuffer = zeros(blockSize, apVast.m_numberOfSrcs);
        end

        %% Main function (public interface)
        function [outputBuffersA, outputBuffersB, obj] = processInputBuffer(obj, inputA, inputB)
            arguments (Input)
                obj (1,1) apVast
                inputA (:, 1)
                inputB (:, 1)
            end
            arguments (Output)
                outputBuffersA (:,:) double
                outputBuffersB (:,:) double
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
                targetAtoASpectra(:,mIdx) = fft(obj.m_window .* obj.m_loudspeakerTargetResponseAtoABuffer(:,mIdx));
                targetBtoBSpectra(:,mIdx) = fft(obj.m_window .* obj.m_loudspeakerTargetResponseBtoBBuffer(:,mIdx));
            end

            obj.updatePerceptualWeighting(targetAtoASpectra, targetBtoBSpectra);

            % Circular convolution with weighting filter
            targetAtoASpectra = targetAtoASpectra .* obj.m_weightingSpectraA;
            targetBtoBSpectra = targetBtoBSpectra .* obj.m_weightingSpectraB;

            % WOLA reconstruction
            for mIdx = 1 : obj.m_numberOfMics
                % Zone A
                tmpOld = obj.m_loudspeakerWeightedTargetResponseAtoAOverlapBuffer(:,mIdx);
                tmpNew = obj.m_window .* ifft(targetAtoASpectra(:,mIdx));
                obj.m_loudspeakerWeightedTargetResponseAtoAOverlapBuffer(:,mIdx) = [tmpOld(obj.m_hopSize+1:obj.m_blockSize); zeros(obj.m_hopSize,1)] + tmpNew;
                % Zone B
                tmpOld = obj.m_loudspeakerWeightedTargetResponseBtoBOverlapBuffer(:,mIdx);
                tmpNew = obj.m_window .* ifft(targetBtoBSpectra(:,mIdx));
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
                AtoASpectra(:,:,mIdx) = fft(repmat(obj.m_window, 1, obj.m_numberOfSrcs) .* obj.m_loudspeakerResponseAtoABuffer(:,:,mIdx));
                AtoBSpectra(:,:,mIdx) = fft(repmat(obj.m_window, 1, obj.m_numberOfSrcs) .* obj.m_loudspeakerResponseAtoBBuffer(:,:,mIdx));
                BtoASpectra(:,:,mIdx) = fft(repmat(obj.m_window, 1, obj.m_numberOfSrcs) .* obj.m_loudspeakerResponseBtoABuffer(:,:,mIdx));
                BtoBSpectra(:,:,mIdx) = fft(repmat(obj.m_window, 1, obj.m_numberOfSrcs) .* obj.m_loudspeakerResponseBtoBBuffer(:,:,mIdx));
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
                tmpNew = repmat(obj.m_window,1,obj.m_numberOfSrcs) .* ifft(AtoASpectra(:,:,mIdx), obj.m_blockSize, 1);
                obj.m_loudspeakerWeightedResponseAtoAOverlapBuffer(:,:,mIdx) = [tmpOld(obj.m_hopSize+1:obj.m_blockSize,:); zeros(obj.m_hopSize,obj.m_numberOfSrcs)] + tmpNew;
                % Signal A to Zone B
                tmpOld = obj.m_loudspeakerWeightedResponseAtoBOverlapBuffer(:,:,mIdx);
                tmpNew = repmat(obj.m_window,1,obj.m_numberOfSrcs) .* ifft(AtoBSpectra(:,:,mIdx), obj.m_blockSize, 1);
                obj.m_loudspeakerWeightedResponseAtoBOverlapBuffer(:,:,mIdx) = [tmpOld(obj.m_hopSize+1:obj.m_blockSize,:); zeros(obj.m_hopSize,obj.m_numberOfSrcs)] + tmpNew;
                % Signal B to Zone A
                tmpOld = obj.m_loudspeakerWeightedResponseBtoAOverlapBuffer(:,:,mIdx);
                tmpNew = repmat(obj.m_window,1,obj.m_numberOfSrcs) .* ifft(BtoASpectra(:,:,mIdx), obj.m_blockSize, 1);
                obj.m_loudspeakerWeightedResponseBtoAOverlapBuffer(:,:,mIdx) = [tmpOld(obj.m_hopSize+1:obj.m_blockSize,:); zeros(obj.m_hopSize,obj.m_numberOfSrcs)] + tmpNew;
                % Signal B to Zone B
                tmpOld = obj.m_loudspeakerWeightedResponseBtoBOverlapBuffer(:,:,mIdx);
                tmpNew = repmat(obj.m_window,1,obj.m_numberOfSrcs) .* ifft(BtoBSpectra(:,:,mIdx), obj.m_blockSize, 1);
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
        end

        function [obj] = updateStatistics(obj)
            arguments (Input)
                obj (1,1) apVast
            end
            arguments (Output)
                obj (1,1) apVast
            end
            obj.resetStatistics();
            for nIdx = 1 : obj.m_statisticsBufferLength - obj.m_filterLength
                idx = nIdx + (0:obj.m_filterLength-1);
                for mIdx = 1 : obj.m_numberOfMics
                    tmp = flipud(obj.m_loudspeakerWeightedResponseAtoABuffer(idx,:,mIdx));
                    yAtoA = tmp(:);
                    tmp = flipud(obj.m_loudspeakerWeightedResponseAtoBBuffer(idx,:,mIdx));
                    yAtoB = tmp(:);
                    tmp = flipud(obj.m_loudspeakerWeightedResponseBtoABuffer(idx,:,mIdx));
                    yBtoA = tmp(:);
                    tmp = flipud(obj.m_loudspeakerWeightedResponseBtoBBuffer(idx,:,mIdx));
                    yBtoB = tmp(:);

                    dA = flipud(obj.m_loudspeakerWeightedTargetResponseAtoABuffer(idx,mIdx));
                    dB = flipud(obj.m_loudspeakerWeightedTargetResponseBtoBBuffer(idx,mIdx));

                    obj.m_RAtoA = obj.m_RAtoA + yAtoA * yAtoA';
                    obj.m_RAtoB = obj.m_RAtoB + yAtoB * yAtoB';
                    obj.m_RBtoA = obj.m_RBtoA + yBtoA * yBtoA';
                    obj.m_RBtoB = obj.m_RBtoB + yBtoB * yBtoB';

                    obj.m_rA = obj.m_rA + yAtoA * dA(1);
                    obj.m_rB = obj.m_rB + yBtoB * dB(1);
                end
            end
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
                numberOfEigenvectors (1,1) double
            end
            arguments (Output)
                obj (1,1) apVast
            end
            % Joint diagonalization
            [UA,lambdaA] = jdiag(obj.m_RAtoA, obj.m_RAtoB, 'vector', true);
            [UB,lambdaB] = jdiag(obj.m_RBtoB, obj.m_RBtoA, 'vector', true);

            % NOTE: alternative, check if working?
            % [UA,lambdaA] = jdiag(obj.m_RAtoA, obj.m_RAtoB);
            % [UB,lambdaB] = jdiag(obj.m_RBtoB, obj.m_RBtoA);
            % lambdaA = diag(lambdaA)
            % lambdaB = diag(lambdaB)

            % Store for debugging
            obj.m_UA = UA 
            obj.m_UB = UB 
            obj.m_lambdaA = lambdaA
            obj.m_lambdaB = lambdaB

            % Determine filters
            obj.m_wA = zeros(obj.m_filterLength*obj.m_numberOfSrcs,1);
            obj.m_wB = zeros(obj.m_filterLength*obj.m_numberOfSrcs,1);
            for i = 1:numberOfEigenvectors
                obj.m_wA = obj.m_wA + (obj.m_UA(:,i).'*obj.m_rA)/(obj.m_lambdaA(i) + mu) * obj.m_UA(:,i);
                obj.m_wB = obj.m_wB + (obj.m_UB(:,i).'*obj.m_rB)/(obj.m_lambdaB(i) + mu) * obj.m_UB(:,i);
            end

            % Determine filter spectra
            obj.m_filterSpectraA = fft(reshape(obj.m_wA, obj.m_filterLength, obj.m_numberOfSrcs), obj.m_blockSize, 1);
            obj.m_filterSpectraB = fft(reshape(obj.m_wB, obj.m_filterLength, obj.m_numberOfSrcs), obj.m_blockSize, 1);
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

        function [outputBuffersA, outputBuffersB, obj] = computeOutputBuffers(obj)
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

            % Circular convolution with the filter spectra
            outputSpectraA = repmat(inputSpectrumA, 1, obj.m_numberOfSrcs) .* obj.m_filterSpectraA;
            outputSpectraB = repmat(inputSpectrumB, 1, obj.m_numberOfSrcs) .* obj.m_filterSpectraB;

            % Update the output overlap buffers
            idx = obj.m_hopSize + 1:obj.m_blockSize;
            obj.m_outputAOverlapBuffer = [obj.m_outputAOverlapBuffer(idx,:); zeros(obj.m_hopSize,obj.m_numberOfSrcs)] + ifft(outputSpectraA, obj.m_blockSize, 1) .* repmat(obj.m_window, 1, obj.m_numberOfSrcs);
            obj.m_outputBOverlapBuffer = [obj.m_outputBOverlapBuffer(idx,:); zeros(obj.m_hopSize,obj.m_numberOfSrcs)] + ifft(outputSpectraB, obj.m_blockSize, 1) .* repmat(obj.m_window, 1, obj.m_numberOfSrcs);

            % Extract samples for the output buffers
            outputBuffersA = obj.m_outputAOverlapBuffer(1:obj.m_hopSize,:);
            outputBuffersB = obj.m_outputBOverlapBuffer(1:obj.m_hopSize,:);
        end

        
  
    end
end
