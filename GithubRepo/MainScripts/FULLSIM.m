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
Rx_Array = (lambda_c/2)*[-2,-1,0,1,2;-1,-1,-1,-1,-1;0,0,0,0,0].';




%% Generate the signal per antenna
% Gold Sequence for each antenna, pre calculated shifts for balanced sequences
goldseq1 = fGoldSeq(M_sequence1, M_sequence2, 5);
goldseq1(goldseq1 == 1) = -1;
goldseq1(goldseq1 == 0) = 1;

goldseq2 = fGoldSeq(M_sequence1, M_sequence2, 6);
goldseq2(goldseq2 == 1) = -1;
goldseq2(goldseq2 == 0) = 1;

goldseq3 = fGoldSeq(M_sequence1, M_sequence2, 7);
goldseq3(goldseq3 == 1) = -1;
goldseq3(goldseq3 == 0) = 1;

goldseq4 = fGoldSeq(M_sequence1, M_sequence2, 9);
goldseq4(goldseq4 == 1) = -1;
goldseq4(goldseq4 == 0) = 1;

goldseq5 = fGoldSeq(M_sequence1, M_sequence2, 13);
goldseq5(goldseq5 == 1) = -1;
goldseq5(goldseq5 == 0) = 1;

% Spreading sequence
goldseq_spread = goldseq1;

% Number of samples for the '1' and '0' segments
N = 15; % Number of samples for the '1' segment
M = 1024 - 15; % Number of samples for the '0' segment

% Initialize array to store spread signals
Spread_Signal_Ant = cell(1, 5);

% Trial amplitude
Trial_amp = 32000;

% Function to generate the spread signal for a given antenna
generateSpreadSignal = @(goldseq) kron(Trial_amp * [goldseq.', zeros(1, M)], goldseq_spread.');

% Generate and normalize signals for each antenna
goldseqs = {goldseq1, goldseq2, goldseq3, goldseq4, goldseq5};
for i = 1:5
    % Signal for the current antenna
    signal_Pn_ant = Trial_amp * [goldseqs{i}.', zeros(1, M)];
    
    % Normalize power to 1
    signal_power = mean(signal_Pn_ant(:).^2);
    signal_Pn_ant = signal_Pn_ant / sqrt(signal_power);
    
    % Spread the signal using goldseq_spread
    Spread_Signal_Ant{i} = kron(signal_Pn_ant, goldseq_spread.');
    
    % Display the mean power to check normalization
    disp(['Antenna ' num2str(i) ' signal power: ' num2str(mean(signal_Pn_ant(:).^2))])
end

% Spread signals for each antenna 
Spread_Signal_Ant1 = Spread_Signal_Ant{1};
Spread_Signal_Ant2 = Spread_Signal_Ant{2};
Spread_Signal_Ant3 = Spread_Signal_Ant{3};
Spread_Signal_Ant4 = Spread_Signal_Ant{4};
Spread_Signal_Ant5 = Spread_Signal_Ant{5};

%% Multiply by carrier
carrier_signal = zeros(1, 15360);
% Generate the carrier signal samples
for n_samples = 1:15360
    t = (n_samples-1)* Ts; % Time instance for each sample
    carrier_signal(n_samples) = exp(1i*2*pi*Fc*t);
end
Full_Tx_signal = [Spread_Signal_Ant1.*carrier_signal;Spread_Signal_Ant2.*carrier_signal;Spread_Signal_Ant3.*carrier_signal;Spread_Signal_Ant4.*carrier_signal;
    Spread_Signal_Ant5.*carrier_signal];
%% Target
% Target 1:
Range1 = 15.5871; %range of the target in meters
delay1 = 230;%Delay in samples (i.e. clock time)
pad_zeros1 = 16227-delay1-15360;
RCS1 = 1;% RCS of the target, assumed 1
theta1 = deg2rad(135);
k_1 = 2*pi*(Fc/c_light)*[cos(theta1)*cos(0), sin(theta1)*cos(0), sin(0)].';
S_Tx_1 = exp(-1i*Tx_array*k_1); % SPV for target

B1 = 1; %sqrt(1/((4*pi)^3))*(lambda_c/(Range1^2))*sqrt(RCS1)*exp(-1i*2*pi*Fc*(2*Range1/c_light));
Target_1_response = B1*(S_Tx_1*S_Tx_1')*Full_Tx_signal;
Target_1_delayed = [zeros(5,delay1),Target_1_response,zeros(5,pad_zeros1)]; %Delayed signal, padded with end zeros for standard length

% Target 1_d:
Range1 = 15.5871; %range of the target in meters
delay1 = 230;%Delay in samples (i.e. clock time)
pad_zeros1 = 16227-delay1-15360;
RCS1 = 1;% RCS of the target, assumed 1
theta1 = deg2rad(56);
k_1 = 2*pi*(Fc/c_light)*[cos(theta1)*cos(0), sin(theta1)*cos(0), sin(0)].';
S_Tx_1 = exp(-1i*Tx_array*k_1); % SPV for target

B1 = 1; %sqrt(1/((4*pi)^3))*(lambda_c/(Range1^2))*sqrt(RCS1)*exp(-1i*2*pi*Fc*(2*Range1/c_light));
Target_1_response = B1*(S_Tx_1*S_Tx_1')*Full_Tx_signal;
Target_1_delayed_d = [zeros(5,delay1),Target_1_response,zeros(5,pad_zeros1)];

% Target 2
Range2 = 26.3625; %range of the target in meters
delay2 = 386;%Delay in samples (i.e. clock time)
pad_zeros2 = 16227-delay2-15360;
RCS2 = 1;% RCS of the target, assumed 1
theta2 = deg2rad(112);
k_2 = 2*pi*(Fc/c_light)*[cos(theta2)*cos(0), sin(theta2)*cos(0), sin(0)].';
S_Tx_2 = exp(-1i*Tx_array*k_2); % SPV for target

B2 = 1;%sqrt(1/((4*pi)^3))*(lambda_c/(Range2^2))*sqrt(1);
Target_2_response = B2*(S_Tx_2*S_Tx_2')*Full_Tx_signal;
Target_2_delayed = [zeros(5,delay2),Target_2_response,zeros(5,pad_zeros2)]; %Delayed signal, padded with end zeros for standard length

% Target 3
Range3 = 37.1379; %range of the target in meters
delay3 = 548;%Delay in samples (i.e. clock time)
pad_zeros3 = 16227-delay3-15360;
RCS3 = 1;% RCS of the target, assumed 1
theta3 = deg2rad(93);
k_3 = 2*pi*(Fc/c_light)*[cos(theta3)*cos(0), sin(theta3)*cos(0), sin(0)].';
S_Tx_3 = exp(-1i*Tx_array*k_3); % SPV for target

B3 = 1;%sqrt(1/((4*pi)^3))*(lambda_c/(Range3^2))*sqrt(1);
Target_3_response = B3*(S_Tx_3*S_Tx_3')*Full_Tx_signal;
Target_3_delayed = [zeros(5,delay3),Target_3_response,zeros(5,pad_zeros3)]; %Delayed signal, padded with end zeros for standard length

%Target 4:
Range4 = 47.913336; %range of the target in meters
delay4 = 707;%Delay in samples (i.e. clock time)
pad_zeros4 = 16227-delay4-15360;
RCS4 = 1;% RCS of the target, assumed 1
theta4 = deg2rad(68);
k_4 = 2*pi*(Fc/c_light)*[cos(theta4)*cos(0), sin(theta4)*cos(0), sin(0)].';
S_Tx_4 = exp(-1i*Tx_array*k_4); % SPV for target

B4 = 1;%sqrt(1/((4*pi)^3))*(lambda_c/(Range4^2))*sqrt(0.5);
Target_4_response = B4*(S_Tx_4*S_Tx_4')*Full_Tx_signal;
Target_4_delayed = [zeros(5,delay4),Target_4_response,zeros(5,pad_zeros4)]; %Delayed signal, padded with end zeros for standard length

% Target 5:
Range5 = 58.6887543; %range of the target in meters
delay5 = 866;%Delay in samples (i.e. clock time)
pad_zeros5 = 16227-delay5-15360;
RCS5 = 1;% RCS of the target, assumed 1
theta5 = deg2rad(45);
k_5 = 2*pi*(Fc/c_light)*[cos(theta5)*cos(0), sin(theta5)*cos(0), sin(0)].';
S_Tx_5 = exp(-1i*Tx_array*k_5); % SPV for target

B5 = 1;
Target_5_response = B5*(S_Tx_5*S_Tx_5')*Full_Tx_signal;
Target_5_delayed = [zeros(5,delay5),Target_5_response,zeros(5,pad_zeros5)]; 

%Target 6

Range5 = 58.6887543; %range of the target in meters
delay6 = 853;%Delay in samples (i.e. clock time)
pad_zeros6 = 16227-delay6-15360;
RCS5 = 1;% RCS of the target, assumed 1
theta6 = deg2rad(96);
k_6 = 2*pi*(Fc/c_light)*[cos(theta6)*cos(0), sin(theta6)*cos(0), sin(0)].';
S_Tx_6 = exp(-1i*Tx_array*k_6); % SPV for target

B6 = 1;
Target_6_response = B6*(S_Tx_6*S_Tx_6')*Full_Tx_signal;
Target_6_delayed = [zeros(5,delay6),Target_6_response,zeros(5,pad_zeros6)]; 



%% Sum of channels + Noise addition

SNR = 40;
Received_signal_nonoise =Target_1_delayed_d+ Target_2_delayed +Target_4_delayed+Target_3_delayed+Target_1_delayed+Target_5_delayed+Target_6_delayed;%noise_matrix;
Received_signal = awgn(Received_signal_nonoise,SNR);
%Received_signal=  Target_2_delayed +Target_4_delayed+Target_3_delayed+Target_1_delayed+Target_5_delayed;



%% Gold Sequence definition - Antenna 0
goldseq = goldseq1;%[-1;1;-1;-1;1;-1;-1;1;-1;1;1;-1;-1;1;1];

%% Multiply by carrier
carrier_signal = zeros(1, 16227);
% Generate the carrier signal samples
for n_samples = 1:16227
    t = (n_samples-1)* Ts; % Time instance for each sample
    carrier_signal(n_samples) = exp(-1i*2*pi*Fc*t);
end
Full_Rx_signal = [Received_signal(1,:).*carrier_signal;Received_signal(2,:).*carrier_signal;Received_signal(3,:).*carrier_signal;Received_signal(4,:).*carrier_signal;
    Received_signal(5,:).*carrier_signal];
%% Form 3D Data Cube:
x_3D = threed_cube(Full_Rx_signal,goldseq);


%% Covariance and Noise Subspace Estimation
lengt = 540;%31578 or 1053
sum=zeros(150,150);
for i = 1:lengt
    term = x_3D(:,i) * x_3D(:,i)';
    sum = sum+term;
end
Rxx_3d = (1/lengt)*sum;

K = fMinDesLength(Rxx_3d,5,1053);
disp(K)
K=7; % 19 works for 2 targets   %For alternate threedcube(the shorter one) 2 works for 2 on imag resluts

% Compute eigenvalues and eigenvectors
[vector, values] = eig(Rxx_3d);

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
Pn = eye(150)-PEs;
%% MUSIC Initialization
%Initialize theta for the loop:

theta_start = 1;
theta_end = 180;
theta_increment = 1;

delay_start = 1;
delay_increment = 1;
delay_end =15;

%Shifting MAtrix:
% Assuming N is the desired size of the shifting matrix
Shift_N = 2*length(goldseq);    %SET THIS BASED ON THE GOLD SEQUENCE LENGTH
I = eye(Shift_N-1);
I_shifted = [zeros(1, Shift_N-1); I];
shifting_matrix = [I_shifted, zeros(Shift_N, 1)];


c =c_light;

%Array Definition
R = Rx_Array ;

% Initialize an empty matrix to store theta values, delay and cost values
results = [];

%% MUSIC Algorithm:

% Loop through the values of theta from 0 to 180 in increments of 1
for theta = theta_start:theta_increment:theta_end
    for delay = delay_start:delay_increment:delay_end

        theta_rad = deg2rad(theta);
        u = 2*pi*(Fc/c)*[cos(theta_rad) * cos(0) , sin(theta_rad)*cos(0), sin(0)].';
        
        % Calculate S
        S = exp(-1i * R * u);
       
        delay_mod = mod((delay),15);

        c_shifted = (shifting_matrix^delay_mod)*[goldseq_spread;zeros(15,1)];

        h = kron(S,c_shifted);
        
        % Calculate the cost value
        %cost_value = 1/(h' * Pn * h);
        
        cost_value = 1/ (h' * (Es*Es') * h);
        %convert to power
        musicdB = 10 * log10(cost_value);
        % Store the theta value and cost value in the matrix
        results = [results; theta,cost_value,delay_mod];
    end
end

%% Plot Results:
x_plot = double(results(:, 1));
y = double(results(:, 3)); 
z = double(results(:, 2)); 

[X, Y] = meshgrid(linspace(min(x_plot), max(x_plot), 100), linspace(min(y), max(y), 100));


Z_mag = griddata(x_plot, y, abs(z), X, Y); % Magnitude of z
Z_phase = griddata(x_plot, y, imag(z), X, Y); % Phase of z
Z_real = griddata(x_plot, y, real(z), X, Y);
Z_db = griddata(x_plot, y, abs(10 * log10(z)), X, Y);



figure(4);
surf(X, Y, Z_db);
xlabel('Theta  (degrees)');
ylabel('Delay (Ts)');
zlabel('Magnitude of Z');
title('Magnitude of DB Value vs. DOA');
grid on;
colorbar; 



% % Plot the results as a surface for magnitude
% figure;
% surf(X, Y, Z_mag);
% xlabel('Theta  (degrees)');
% ylabel('Delay (Ts)');
% zlabel('Magnitude of Z');
% title('Magnitude of ABS Cost Value vs. DOA');
% grid on;
% colorbar; 
% 
% % Plot the results as a surface for magnitude
% figure(2);
% surf(X, Y, Z_phase);
% xlabel('Theta  (degrees)');
% ylabel('Delay (Ts)');
% zlabel('Magnitude of Z');
% title('Magnitude of IMAG Value vs. DOA');
% grid on;
% colorbar; 
% 
% % Plot the results as a surface for magnitude
% figure(3);
% surf(X, Y, Z_real);
% xlabel('Theta  (degrees)');
% ylabel('Delay (Ts)');
% zlabel('Magnitude of Z');
% title('Magnitude of Real Value vs. DOA');
% grid on;
% colorbar; 


