%% Set up constants for Simulation:
Fc =2400e6;
c_light = 299792458;
Fs = 2211.84e6;
Ts = 1/Fs;
lambda_c = c_light/Fc;


% M Sequences for modulation:
M_sequence1 = fMSeqGen([1;0;0;1;1]).';
M_sequence2 = fMSeqGen([1;1;0;0;1]).';


N_samples = 1024;



%% Generate the signal per antenna
% Gold Sequence for each antenna, pre calculated shifts for balanced sequences
goldseq1 = fGoldSeq(M_sequence1, M_sequence2, 7);
goldseq1(goldseq1 == 1) = -1;
goldseq1(goldseq1 == 0) = 1;

goldseq2 = fGoldSeq(M_sequence1, M_sequence2, 6);
goldseq2(goldseq2 == 1) = -1;
goldseq2(goldseq2 == 0) = 1;

goldseq_spread = fGoldSeq(M_sequence1, M_sequence2, 5);
goldseq_spread(goldseq_spread == 1) = -1;
goldseq_spread(goldseq_spread == 0) = 1;





%% Generate Carrier Signal Samples:
% Initialize the carrier signal array
carrier_signal = zeros(1, 15360);

% Generate the carrier signal samples
for n = 1:15360
    t = (n-1)* Ts; % Time instance for each sample
    carrier_signal(n) = exp(-1i*2*pi*Fc*t);
end





%% Generate Transmitted signal:
% Signal for the Antenna 1
M = N_samples-15;
signal_Pn_ant = 8000 * [goldseq1.', zeros(1, M)];



% Spread the signal using goldseq_spread
Spread_Signal_Ant1 = kron(signal_Pn_ant, goldseq_spread.');

%Ant 2
signal_Pn_ant = 8000 * [goldseq2.', zeros(1, M)];



% Spread the signal using goldseq_spread
Spread_Signal_Ant2 = kron(signal_Pn_ant, goldseq_spread.');
    






 %% Write signal to TDMS file - Real
%Create signal table
signal = Spread_Signal_Ant1.*carrier_signal;
%outsig = signal_Pn;
%plot(outsig(1:400))
Cutsignal = signal(1:5120);
S_table = array2table(real(Cutsignal.'), 'VariableNames', {'Real'});



% Write it for 1 PRI
tdmswrite("FinalREALsig_fullmod_Ant1.tdms", {S_table}, ChannelGroupNames= "Real");


%Create signal table
signal2 = Spread_Signal_Ant2.*carrier_signal;
%outsig = signal_Pn;
%plot(outsig(1:400))
Cutsignal2 = signal2(1:5120);
S_table2 = array2table(real(Cutsignal2.'), 'VariableNames', {'Real'});



% Write it for 1 PRI
tdmswrite("FinalREALsig_fullmod_Ant2.tdms", {S_table2}, ChannelGroupNames= "Real");




