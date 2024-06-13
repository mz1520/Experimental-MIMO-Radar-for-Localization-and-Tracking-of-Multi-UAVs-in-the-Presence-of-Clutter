%% Set up constants for Simulation:
Fc =2400e6;
c_light = 299792458;
Fs = 2211.84e6;
Ts = 1/Fs;
lambda_c = c_light/Fc;


% M Sequences for modulation:
M_sequence1 = fMSeqGen([1;0;0;1;1]).';
M_sequence2 = fMSeqGen([1;1;0;0;1]).';

% Arrays Both Linear, Collocated:
Tx_array = (lambda_c/2)*[-2,-1,0,1,2;0,0,0,0,0;0,0,0,0,0].';
Rx_Array = (lambda_c/2)*[-2,-1,0,1,2;0,0,0,0,0;0,0,0,0,0].';





%% Generate the transmitted signal per antenna:
% Gold Sequence for each antenna, pre calculated shifts for balanced
% sequences
goldseq1 = fGoldSeq(M_sequence1,M_sequence2,5);
goldseq1(goldseq1 == 1) = -1;
goldseq1(goldseq1 == 0) = 1;

goldseq2 = fGoldSeq(M_sequence1,M_sequence2,6);
goldseq2(goldseq2 == 1) = -1;
goldseq2(goldseq2 == 0) = 1;

goldseq3 = fGoldSeq(M_sequence1,M_sequence2,7);
goldseq3(goldseq3 == 1) = -1;
goldseq3(goldseq3 == 0) = 1;

goldseq4 = fGoldSeq(M_sequence1,M_sequence2,9);
goldseq4(goldseq4 == 1) = -1;
goldseq4(goldseq4 == 0) = 1;

goldseq5 = fGoldSeq(M_sequence1,M_sequence2,13);
goldseq5(goldseq5 == 1) = -1;
goldseq5(goldseq5 == 0) = 1;

%Antenna 1:

N = 15; % Number of samples for the '1' segment
M = 1024 - 15;  % Number of samples for the '0' segment
% Generate the signal
signal_ones_Pn = goldseq1.';%Segment of 1's by the goldseq
signal_ones = ones(1, N);% Segment of 1's
signal_zeros = zeros(1, M);  % Segment of 0's

Trial_amp =1;

% Concatenate the segments to form the final signal
signal_Pn_ant1 = Trial_amp*[signal_ones_Pn,signal_zeros];
%normalize Power to 1
signal_power = mean(signal_Pn_ant1(:).^2);
disp(signal_power)
%signal_Pn_ant1 = signal_Pn_ant1 / sqrt(signal_power);

%signal_Pn_ant5 = signal_Pn_ant5 / sqrt(signal_power);

Full_Tx_signal_unmodulated = [signal_Pn_ant1;signal_Pn_ant1;signal_Pn_ant1;signal_Pn_ant1;signal_Pn_ant1];
%% Multiply by carrier
carrier_signal = zeros(1, 1024);
% Generate the carrier signal samples
for n_samples = 1:1024
    t = (n_samples-1)* Ts; % Time instance for each sample
    carrier_signal(n_samples) = exp(1i*2*pi*Fc*t);
end
Full_Tx_signal = [signal_Pn_ant1.*carrier_signal;signal_Pn_ant1.*carrier_signal;signal_Pn_ant1.*carrier_signal;signal_Pn_ant1.*carrier_signal;
    signal_Pn_ant1.*carrier_signal];
%% Target
% Target 1:
Range1 = 15.5871; %range of the target in meters
delay1 = 230;%Delay in samples (i.e. clock time)
pad_zeros1 = 1891-delay1-1024;
RCS1 = 1;% RCS of the target, assumed 1
theta1 = deg2rad(135);
k_1 = 2*pi*(Fc/c_light)*[cos(theta1)*cos(0), sin(theta1)*cos(0), sin(0)].';
S_Tx_1 = exp(-1i*Tx_array*k_1); % SPV for target

B1 = 1; %sqrt(1/((4*pi)^3))*(lambda_c/(Range1^2))*sqrt(RCS1)*exp(-1i*2*pi*Fc*(2*Range1/c_light));
Target_1_response = B1*(S_Tx_1*S_Tx_1')*Full_Tx_signal;
Target_1_delayed = [zeros(5,delay1),Target_1_response,zeros(5,pad_zeros1)]; %Delayed signal, padded with end zeros for standard length

% Target 2
Range2 = 26.3625; %range of the target in meters
delay2 = 389;%Delay in samples (i.e. clock time)
pad_zeros2 = 1891-delay2-1024;
RCS2 = 1;% RCS of the target, assumed 1
theta2 = deg2rad(112);
k_2 = 2*pi*(Fc/c_light)*[cos(theta2)*cos(0), sin(theta2)*cos(0), sin(0)].';
S_Tx_2 = exp(-1i*Tx_array*k_2); % SPV for target

B2 = 1;
Target_2_response = B2*(S_Tx_2*S_Tx_2')*Full_Tx_signal;
Target_2_delayed = [zeros(5,delay2),Target_2_response,zeros(5,pad_zeros2)]; %Delayed signal, padded with end zeros for standard length

% Target 3
Range3 = 37.1379; %range of the target in meters
delay3 = 548;%Delay in samples (i.e. clock time)
pad_zeros3 = 1891-delay3-1024;
RCS3 = 1;% RCS of the target, assumed 1
theta3 = deg2rad(93);
k_3 = 2*pi*(Fc/c_light)*[cos(theta3)*cos(0), sin(theta3)*cos(0), sin(0)].';
S_Tx_3 = exp(-1i*Tx_array*k_3); % SPV for target

B3 = 1;
Target_3_response = B3*(S_Tx_3*S_Tx_3')*Full_Tx_signal;
Target_3_delayed = [zeros(5,delay3),Target_3_response,zeros(5,pad_zeros3)]; %Delayed signal, padded with end zeros for standard length

%Target 4:
Range4 = 47.913336; %range of the target in meters
delay4 = 707;%Delay in samples (i.e. clock time)
pad_zeros4 = 1891-delay4-1024;
RCS4 = 1;% RCS of the target, assumed 1
theta4 = deg2rad(68);
k_4 = 2*pi*(Fc/c_light)*[cos(theta4)*cos(0), sin(theta4)*cos(0), sin(0)].';
S_Tx_4 = exp(-1i*Tx_array*k_4); % SPV for target

B4 = 1;
Target_4_response = B4*(S_Tx_4*S_Tx_4')*Full_Tx_signal;
Target_4_delayed = [zeros(5,delay4),Target_4_response,zeros(5,pad_zeros4)]; %Delayed signal, padded with end zeros for standard length

% Target 5:
Range5 = 58.6887543; %range of the target in meters
delay5 = 866;%Delay in samples (i.e. clock time)
pad_zeros5 = 1891-delay5-1024;
RCS5 = 1;% RCS of the target, assumed 1
theta5 = deg2rad(45);
k_5 = 2*pi*(Fc/c_light)*[cos(theta5)*cos(0), sin(theta5)*cos(0), sin(0)].';
S_Tx_5 = exp(-1i*Tx_array*k_5); % SPV for target

B5 = 1;
Target_5_response = B5*(S_Tx_5*S_Tx_5')*Full_Tx_signal;
Target_5_delayed = [zeros(5,delay5),Target_5_response,zeros(5,pad_zeros5)]; 



%% Sum of channels + Noise addition
sigma_square = 0.0001;
sigma = sqrt(sigma_square);
% Generate the noise matrix
noise_matrix = sigma * randn(5, 1891);
Rnn = cov(noise_matrix.');
SNR = 40;




Received_signal_nonoise = Target_1_delayed + Target_2_delayed+Target_3_delayed+ Target_4_delayed+Target_5_delayed;%noise_matrix;
Received_signal = awgn(Received_signal_nonoise,SNR);


%% Multily By carrier
%% Multiply by carrier
carrier_signal = zeros(1, 1891);
% Generate the carrier signal samples
for n_samples = 1:1891
    t = (n_samples-1)* Ts; % Time instance for each sample
    carrier_signal(n_samples) = exp(-1i*2*pi*Fc*t);
end
Full_Rx_signal = [Received_signal(1,:).*carrier_signal;Received_signal(2,:).*carrier_signal;Received_signal(3,:).*carrier_signal;Received_signal(4,:).*carrier_signal;
    Received_signal(5,:).*carrier_signal];

%% Form 3D Data Cube:


%% Covariance and Noise Subspace Estimation
lengt = 1891;
sum=zeros(5,5);
for i = 1:1891
    term = Full_Rx_signal(:,i) * Full_Rx_signal(:,i)';
    sum = sum+term;
end
Rxx = (1/lengt)*sum;


K = fMinDesLength(Rxx,5,1891);
disp(K)
K=4;

% Compute eigenvalues and eigenvectors
[vector, values] = eig(Rxx);

% Extract the eigenvalues from the diagonal of the values matrix
eigenvalues = diag(values);

% Sort the eigenvalues in descending order and get the indices
[sorted_eigenvalues, sorted_indices] = sort(eigenvalues, 'ascend');%meant to be descend

% Select the top K eigenvalues and their corresponding indices
topK_indices = sorted_indices(1:K);

% Form Es with the eigenvectors corresponding to the top K eigenvalues
Es = vector(:, topK_indices);

%Projection operator for Es
PEs = Es*inv(Es'*Es)*Es';

%Noise subspace Projection:
Pn = eye(5)-PEs;
%% MUSIC Initialization
%Initialize theta for the loop:

theta_start = 0;
theta_end = 180;
theta_increment = 1;



c =c_light;

%Array Definition
R = Rx_Array ;

% Initialize an empty matrix to store theta values, delay and cost values
results = [];

%% MUSIC Algorithm:

% Loop through the values of theta from 0 to 180 in increments of 1
for theta = theta_start:theta_increment:theta_end
    

    theta_rad = deg2rad(theta);
    u = 2*pi*(Fc/c)*[cos(theta_rad) * cos(0) , sin(theta_rad)*cos(0), sin(0)].';
    
    % Calculate S
    S = exp(-1i * R * u);
  
    
    % Calculate the cost value
   % cost_value = 1/(S' * Pn * S);
    
    cost_value = 1/ (S' * (Es*Es') * S);
    %convert to power
    musicdB = 10 * log10(cost_value);
    % Store the theta value and cost value in the matrix
    results = [results; theta,musicdB];
   
end
% Define the x_plot values
x_plot = 1:181;

% Assume 'results' is already defined with at least 2 columns
% Plot the cost function values
figure;
plot(x_plot, results(:,2), 'b', 'DisplayName', 'Cost Function Value');
hold on;

% Define the x indexes for vertical lines
x_indexes = [135, 112, 93, 68, 45];

% Plot vertical red dashed lines at specified x indexes
for k = 1:length(x_indexes)
    xline(x_indexes(k), 'r--', 'DisplayName', 'Real Target Values');
end

% Customize the legend
lgd=legend('Cost Function Value', 'Real Target Values');


% Increase the font size of the legend
set(lgd, 'FontSize', 18);

% Add labels and title with increased font sizes
xlabel('$\theta$ (Degrees)', 'Interpreter', 'latex', 'FontSize', 24);
ylabel('Cost Function Values (dB)', 'FontSize', 24);
title('Basic MUSIC DOA Estimation', 'FontSize',24);

% Increase the font size of the axes ticks
set(gca, 'FontSize', 15);

% Hold off to stop adding to the current plot
hold off;
