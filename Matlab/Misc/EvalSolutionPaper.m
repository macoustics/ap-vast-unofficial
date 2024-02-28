function EvalSolutionPaper(gB, gD, h, Fs)
% Implementation of the method described in Simon Galvez, Marcos F. "Time
% Domain Optimization of Filters Used in a Loudspeaker Array for Personal
% Audio" IEEE/ACM Transactions on Audio, Speech, and Language processing,
% Vol. 23, No. 11, November 2015
%
% Input parameters:
% -----------------------
% gB:           Matrix (ndim = 3)
%               Impulse responses from each source to each microphone in the bright zone. Size = [Nb, J, M];
%               Nb: Number of microphones in the bright zone
%               J: Length of impulse responses
%               M: Number of sources
% gD:           Matrix (ndim = 3)
%               Impulse responses from each source to each microphone in
%               the dark zone. Size = [Nd, J, M];
%               Nd: Number of microphones in the dark zone
% h             Matrix (ndim = 2)
%               Filters for sound field control. Size = [M, I];
%

%% Initialize parameters
[Nb, J, M] = size(gB);
Nd = size(gD,1);
% I = size(h,2); % Number of loudspeakers, Filter length
I = 4096*4;

%% Convolve filters with measurements
pBF = zeros(J+I-1, Nb); % Time responses in bright zone
pDF = zeros(J+I-1, Nd); % Time responses in dark zone

PB = zeros(I, Nb); % Frequency response in bright zone
PD = zeros(I, Nd); % Frequency response in dark zone
for m = 1:M
    for n = 1:Nb
%         tmp = conv(squeeze(gB(n,:,m)), h(m,:));
        tmp = 1/(J+I-1)*fft(gB(n,:,m), J+I-1);
        tmpFilter = 1/(J+I-1)*fft(h(m,:),J+I-1);
        pBF(:, n) = pBF(:, n) + (tmp.*tmpFilter).';
        tmp = I/(I)*fft(squeeze(gB(n,:,m)),I);
        tmpFilter = I/(I)*fft(h(m,:),I);
        PB(:,n) = PB(:,n) + (tmp.*tmpFilter).';
    end
    for n = 1:Nd
%         tmp = conv(squeeze(gD(n,:,m)), h(m,:));
        tmp = 1/(J+I-1)*fft(gD(n,:,m), J+I-1);
        tmpFilter = 1/(J+I-1)*fft(h(m,:),J+I-1);
        pDF(:, n) = pDF(:,n) + (tmp.*tmpFilter).';
        tmp = I/(I)*fft(squeeze(gD(n,:,m)),I);
        tmpFilter = I/(I)*fft(h(m,:),I);
        PD(:,n) = PD(:,n) + (tmp.*tmpFilter).';
    end
end

% pBF = 1/(J+I-1)*fft(pB, J+I-1, 1);
% pDF = 1/(J+I-1)*fft(pD, J+I-1, 1);

%% Calculate performance
meanBF = mean(abs(pBF).^2, 2);
meanB = mean(abs(PB).^2, 2);
meanDF = mean(abs(pDF).^2, 2);
meanD = mean(abs(PD).^2, 2);

SPL = mean(10*log10( abs(PB).^2/(20e-6)^2 ), 2);
SPLF = mean(10*log10( abs(pBF).^2/(20e-6)^2 ), 2);

Contrast = 10*log10(meanB./meanD);
ContrastF = 10*log10(meanBF./meanDF);
%% Plot the results
FreqF = 0:Fs/(J+I-1):Fs - Fs/(J+I-1);
Freq = 0:Fs/(I):Fs - Fs/(I);
figure(80)

semilogx(Freq, Contrast, 'LineWidth', 1); hold all; grid on
xlim([20 500])
ylim([-10 40])
set(gca,'FontSize',8)
set(gca, 'XTick', [20, 50, 100, 200, 500])
xlabel('Frequency [Hz]'); ylabel('Contrast [dB]');


figure(81)
semilogx(Freq, SPL, 'LineWidth', 1); hold all; grid on
xlim([20 500])
ylim([30 80])
set(gca,'FontSize',8)
set(gca, 'XTick', [20, 50, 100, 200, 500])
xlabel('Frequency [Hz]'); ylabel('SPL [dB]');
% title('Spectrum multiplication')

figure(82)
col = get(groot,'DefaultAxesColorOrder');
for n = 1:8
%     subplot(2,4,n)
%     figure
    plot(h(n,:),'Color',[col(1,:)]); hold on; grid on
    xlim([0 size(h,2)]);
    ylim([-1 1])
    xlabel('Sample number')
end

% figure
% plot(mean(h,1),'Color',[col(1,:)]);
% xlim([0 size(h,2)])
% ylim([-1 1])
% xlabel('Sample number')