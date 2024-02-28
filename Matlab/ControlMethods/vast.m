function [w] = vast(gB, gD, filterLength, ModellingDelay, ReferenceIndex, numberOfEigenvectors, mu)
% Implementation of the method described in T. Lee, et al. "A unified
% approach to generating sound zones using variable span linear filters",
% ICASSP 2018.
%
% Input parameters:
% -----------------------
% gB:           Matrix (ndim = 3)
%               Impulse responses from each source to each microphone in 
%               the bright zone. Size = [numberOfMics, rirLength, numberOfSrc];
%               Nb: Number of microphones in the bright zone
%               I: Length of impulse responses
%               L: Number of sources
% gD:           Matrix (ndim = 3)
%               Impulse responses from each source to each microphone in
%               the dark zone. Size = [numberOfMics, rirLength, numberOfSrc];
%               Nd: Number of microphones in the dark zone
% filterLength  Scalar
%               Length of FIR filter for each loudspeaker
% ModellingDelay    Integer
%               Introduced modelling delay in samples.
% ReferenceIndex:   Integer in [1,L]
%               Index for the reference loudspeaker used to define the
%               target audio signal in the bright zone.
% numberOfEigenvectors   Integer
%               Number of eigenvectors from the joint diagonalization,
%               which are used to approximate the solution. If mu = 1,
%               numberOfEigenvectors = 1 corresponds to the BACC solution
%               and numberOfEigenvectors = filterLength*L corresponds to
%               pressure matching.
% mu            Scalar \in [0,1]
%               Adjustment parameter for the variable span linear filters
%
%
% Output parameters:
% ----------------------
% w:            Matrix (ndim = 2)
%               Determined FIR filters for each loudspeaker. Size = [filterLength, numberOfSrc]
%               [w1(1); ...; w1(filterLength), ..., wM(1); ...; wM(filterLength)];


%% Initialize convolution matrix
tic
[numberOfMics, rirLength, numberOfSrc] = size(gB);

% Determine cross-correlation matrices
RB = zeros(filterLength*numberOfSrc);
RD = RB;
rB = zeros(filterLength*numberOfSrc,1);
N = 1000;
x = [1; zeros(N-1,1)];
xPrev = zeros(rirLength-1,1);
xPad = [xPrev; x];
X = zeros(filterLength, rirLength);
for nIdx = 0:N-1
    xTmp = flip(xPad(nIdx + (1:rirLength)));
    X(2:filterLength,1:rirLength) = X(1:filterLength-1,1:rirLength);
    X(1,1:rirLength) = xTmp;
    for mIdx = 1:numberOfMics
        % Target response
        d = X*[zeros(ModellingDelay,1); squeeze(gB(mIdx,1:end-ModellingDelay,ReferenceIndex))'];
        yB = zeros(filterLength*numberOfSrc,1);
        yD = yB;
        for sIdx = 0:numberOfSrc-1
            idx = sIdx*filterLength + (1:filterLength);
             yB(idx) = X*squeeze(gB(mIdx,:,sIdx+1))';
             yD(idx) = X*squeeze(gD(mIdx,:,sIdx+1))';
        end
        RB = RB + yB*yB';
        
        RD = RD + yD*yD';
        rB = rB + yB * d(1);
    end
end
RB = RB/(numberOfMics*(rirLength-filterLength));
RD = RD/(numberOfMics*(rirLength-filterLength));
rB = rB/(numberOfMics*(rirLength-filterLength));

% keyboard
disp([num2str(toc,'%.4f') 's to form RB, RD, and rB'])


%% Calculate filters
tic
[U,a] = jdiag(RB, RD, 'vector', true);
% a = diag(a);
% keyboard
w = zeros(filterLength*numberOfSrc,1);
for i = 1:numberOfEigenvectors
    w = w + (U(:,i).'*rB)/(a(i) + mu) * U(:,i);
end
% w = (RB + RD)\rB;
disp([num2str(toc,'%.4f') 's to calculate w'])

% keyboard

%% Rearange filters
w = reshape(w,filterLength,numberOfSrc);