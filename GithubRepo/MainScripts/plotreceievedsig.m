
% Data and plot setup
x_plot = 1:5120; %Plots receieved signal in this interval


% Plot the data
figure;
plot(x_plot, xoo(:,1:5120));
hold on;

% Axis labels and title with font size settings
xlabel('Time (Ts)', 'FontSize', 18);
ylabel('Amlitude of Received Signal (LSB)', 'FontSize', 18); 
title('Received Signal Time Domain', 'FontSize', 20); 

% Add a vertical red dashed line at a chosen index
chosen_x_index = 0; 
h=line([chosen_x_index chosen_x_index], ylim, 'Color', 'red', 'LineStyle', '--', 'LineWidth', 2);

legend(h,'Manually Estimated Delay', 'FontSize', 12);


hold off;