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
disp("Creating object OK")

iA = randn(hopSize, 1);
iB = randn(hopSize, 1);
disp("Running...")
[oA, oB] = ap.processInputBuffer(iA, iB);
disp("Running OK")

disp("Saving...")
save("test.mat")
disp("Saving OK")
