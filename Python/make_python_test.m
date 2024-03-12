addpath('../Matlab/ControlMethods');

% TODO: Probably we should be using Martin's RIRs directly, but for now this will suffice
load("rirs.mat")

rirLength = size(rirB, 1);
blockSize = rirLength * 2;
filterLength = 100;
modelingDelay = 20;
referenceIndexA = 7;
referenceIndexB = 7;
numberOfEigenVectors = 50;
mu = 1.0;
statisticsBufferLength = 1000;
hopSize = blockSize / 2;

disp("Creating object...")
ap = apVast(blockSize, rirA, rirB, filterLength, modelingDelay, referenceIndexA, referenceIndexB, numberOfEigenVectors, mu, statisticsBufferLength);
props = properties(ap);
for i = 1:length(props)
    propName = props{i};
    propValue = ap.(propName);
    eval(['before_' propName ' = propValue;']);
end
disp("Creating object OK")

niter = 10;
% iAb = randn(niter, hopSize, 1);
% iBb = randn(niter, hopSize, 1);

load("signals.mat");

sA = signalA(1:niter*hopSize);
sB = signalB(1:niter*hopSize);
iAb = reshape(sA, niter, hopSize);
iBb = reshape(sB, niter, hopSize);

numberOfLoudspeakers = size(rirA, 2);
oAb = zeros(niter, hopSize, numberOfLoudspeakers);
oBb = zeros(niter, hopSize, numberOfLoudspeakers);
wAb = zeros(niter, filterLength * numberOfLoudspeakers);
wBb = zeros(niter, filterLength * numberOfLoudspeakers);
disp("Running...")
for i = 1:niter
    iA = iAb(i,:,:);
    iB = iBb(i,:,:);
    [oA, oB] = ap.processInputBuffer(iA(:), iB(:));
    wAb(i,:) = ap.m_wA;
    wBb(i,:) = ap.m_wB;
    disp(wAb(i, 1:4, 1))
    disp(wBb(i, 1:4, 1))
    oAb(i,:,:) = oA;
    oBb(i,:,:) = oB;
end
props = properties(ap);
for i = 1:length(props)
    propName = props{i};
    propValue = ap.(propName);
    eval(['after_' propName ' = propValue;']);
end
disp("Running OK")

disp("Saving...")
save("test.mat")
disp("Saving OK")
