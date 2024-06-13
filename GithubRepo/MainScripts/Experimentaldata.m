%%Desciption:
%This script uses the data from the ZCU216's ADCs as input to the STAR
%receiver. Must run readadcfiles.mat first to load data into workspace.

%% Receive signals:
%Calling x1 Tile 1 ADC0 antenna:
x1_Im = datasig.ADC_Tile_0___ADC_2.Q.data;
x1_Re = datasig.ADC_Tile_0___ADC_2.I.data;

%X0;
x0_Im = datasig.ADC_Tile_0___ADC_0.Q.data;
x0_Re = datasig.ADC_Tile_0___ADC_0.I.data;

x0_Re_doub = double(x0_Re);
x1_Re_doub = double(x1_Re);

x0_Im_doub = double(x0_Im);
x1_Im_doub = double(x1_Im);



%% Unshift the signals with the phases given in the RFDC

% 
% unshifted_signal0 = x0 .* exp(-1i * Shift_tile0);
% Shift_tile0 = deg2rad(-151.093);
Shift_tile0 = deg2rad(-251.327);
Shift_tile1 = deg2rad(-197.262);

%Apply the phase shift to x0
[theta_x0,rho_x0] = cart2pol(x0_Re_doub,x0_Im_doub);
theta_x0_shift = theta_x0+Shift_tile0;
[x0_shifted_Re,x0_shifted_Im] = pol2cart(theta_x0_shift,rho_x0);
unshifted_signal0 = complex(x0_shifted_Re,x0_shifted_Im);

%Apply the phase shift to x1:
[theta_x1,rho_x1] = cart2pol(x1_Re_doub,x1_Im_doub);
theta_x1_shift = theta_x1+Shift_tile1;
[x1_shifted_Re,x1_shifted_Im] = pol2cart(theta_x1_shift,rho_x1);
unshifted_signal1 = complex(x1_shifted_Re,x1_shifted_Im);




%%
%Form received x vector:
x_int = [unshifted_signal0;unshifted_signal1];
xoo = double(x_int);



%% Set up constants for Simulation:
Fc =2400e6;
c_light = 299792458;
Fs = 2211.84e6;
Ts = 1/Fs;
lambda_c = c_light/Fc;


% M Sequences for modulation:
M_sequence1 = fMSeqGen([1;0;0;1;1]).';
M_sequence2 = fMSeqGen([1;1;0;0;1]).';






%% Generate the signal per antenna
% Gold Sequence for each antenna, pre calculated shifts for balanced sequences
goldseq1 = fGoldSeq(M_sequence1, M_sequence2, 5);
goldseq1(goldseq1 == 1) = -1;
goldseq1(goldseq1 == 0) = 1;


% Spreading sequence
goldseq_spread = goldseq1;
goldseq = goldseq_spread;



%% Multiply by carrier
carrier_signal = zeros(1, 5120);
% Generate the carrier signal samples
for n_samples = 1:5120
    t = (n_samples-1)* Ts; % Time instance for each sample
    carrier_signal(n_samples) = exp(-1i*2*pi*Fc*t);
end
Full_Rx_signal = [xoo(1,:).*carrier_signal;xoo(2,:).*carrier_signal];
%% Form 3D Data Cube:
x_3D = threed_cube(Full_Rx_signal,goldseq_spread);


%% Covariance and Noise Subspace Estimation
lengt = 170;
sum=zeros(60,60);
for i = 1:lengt
    term = x_3D(:,i) * x_3D(:,i)';
    sum = sum+term;
end
Rxx_3d = (1/lengt)*sum;

K = fMinDesLength(Rxx_3d,5,1053);
disp(K)
K=1; % 
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
R = [0.5,-0.5;0,0;0,0].' ;

% Initialize an empty matrix to store theta values, delay and cost values
results = [];

%% MUSIC Algorithm:

% Loop through the values of theta from 0 to 180 in increments of 1
for theta = theta_start:theta_increment:theta_end
    for delay = delay_start:delay_increment:delay_end

        theta_rad = deg2rad(theta);
        u = pi*[cos(theta_rad) * cos(0) , sin(theta_rad)*cos(0), sin(0)].';
        
        % Calculate S
        S = exp(-1i * R * u);
       
        delay_mod = mod((delay),15);

        c_shifted = (shifting_matrix^delay_mod)*[goldseq_spread;zeros(15,1)];

        h = kron(S,c_shifted);
        
        % Calculate the cost value
        %cost_value = 1/(h' * Pn * h);
        
        cost_value = 1/ (h' * (Es*Es') * h);
        
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

% Plot the results as a surface for magnitude
figure;
surf(X, Y, Z_mag);
xlabel('Theta  (degrees)');
ylabel('Delay (Ts)');
zlabel('Magnitude of Z');
title('Magnitude of ABS Cost Value vs. DOA');
grid on;
colorbar; 

% Plot the results as a surface for magnitude
figure(2);
surf(X, Y, Z_phase);
xlabel('Theta  (degrees)');
ylabel('Delay (Ts)');
zlabel('Magnitude of Z');
title('Magnitude of IMAG Value vs. DOA');
grid on;
colorbar; 

% Plot the results as a surface for magnitude
figure(3);
surf(X, Y, Z_real);
xlabel('Theta  (degrees)');
ylabel('Delay (Ts)');
zlabel('Magnitude of Z');
title('Magnitude of Real Value vs. DOA');
grid on;
colorbar; 

figure(4);
surf(X, Y, Z_db);
xlabel('Theta  (degrees)');
ylabel('Delay (Ts)');
zlabel('Magnitude of Z');
title('Magnitude of DB Value vs. DOA');
grid on;
colorbar; 


