function predictedPressure = predictPressure(loudspeakerSignals, rirs)
    % loudspeakerSignals    2D array
    %                       size = [signalLength, numberOfSrcs]
    % rirs                  3D array
    %                       size = [rirLength, numberOfSrcs, numberOfMics]
    % predictedPressure     2D array
    %                       size = [signalLength, numberOfMics]
    signalLength = size(loudspeakerSignals,1);
    numberOfSrcs = size(loudspeakerSignals,2);
    numberOfMics = size(rirs,3);

    predictedPressure = zeros(signalLength, numberOfMics);
    for mIdx = 1:numberOfMics
        for sIdx = 1:numberOfSrcs
            predictedPressure(:,mIdx) = predictedPressure(:,mIdx) + filter(rirs(:,sIdx,mIdx), 1, loudspeakerSignals(:,sIdx));
        end
    end
end