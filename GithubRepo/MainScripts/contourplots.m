% Define the fixed x and y values where the real value's red x's will be plotted
fixed_points = [150,7];

% Plot Results
x_plot = double(results(:, 1));
y = double(results(:, 3)); 
z = double(results(:, 2)); 

% Create a grid of unique x and y values
[X, Y] = meshgrid(linspace(min(x_plot), max(x_plot), 100), linspace(min(y), max(y), 100));


Z_mag = griddata(x_plot, y, abs(z), X, Y); % Magnitude of z
Z_phase = griddata(x_plot, y, imag(z), X, Y); % Phase of z
Z_real = griddata(x_plot, y, real(z), X, Y);
Z_db = griddata(x_plot, y, abs(10 * log10(z)), X, Y);

figure(4);
contour(X, Y, Z_mag, 'ShowText', 'off'); % Disable number labels
xlabel('$\theta$ (Degrees)', 'Interpreter', 'latex', 'FontSize', 23);
ylabel('$\ell (T_c)$', 'Interpreter', 'latex', 'FontSize', 23);
title('STAR Receiver MUSIC Algorithm for Joint DOA, Delay', 'FontSize', 23);
grid on;

hold on;

% Add red x's at fixed [x, y] values
for i = 1:size(fixed_points, 1)
    x_val = fixed_points(i, 1);
    y_val = fixed_points(i, 2);
    plot(x_val, y_val, 'rx', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Real Target Values');
end

% Add legend
legend({'Cost Function Values', 'Real Target Values'}, 'Location', 'best', 'FontSize', 20);

hold off;
