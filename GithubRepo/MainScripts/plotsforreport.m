% Define the fixed x and y values for real target values
fixed_points = [150, 7;]; 

% Plot Results
x_plot = double(results(:, 1));
y = double(results(:, 3)); 
z = double(results(:, 2)); 

[X, Y] = meshgrid(linspace(min(x_plot), max(x_plot), 100), linspace(min(y), max(y), 100));


Z_mag = griddata(x_plot, y, abs(z), X, Y); % Magnitude of z
z_max = max(Z_mag(:));
Z_mag = Z_mag./z_max;
Z_phase = griddata(x_plot, y, imag(z), X, Y); % Phase of z
Z_real = griddata(x_plot, y, real(z), X, Y);
Z_db = griddata(x_plot, y, abs(10 * log10(z)), X, Y);

figure(4);
surf(X, Y, Z_mag);
xlabel('$\theta$ (Degrees)', 'Interpreter', 'latex', 'FontSize', 23);
ylabel('$\ell (T_c)$', 'Interpreter', 'latex', 'FontSize', 23);
zlabel('Cost Function Value (Normalized)', 'FontSize', 23);
title('STAR Receiver MUSIC Algorithm for Joint DOA, Delay', 'FontSize', 23);
grid on;

hold on;

% Add vertical red dashed lines at fixed [x, y] values
z_max = max(Z_mag(:));
line_length = z_max + 0.5; 

for i = 1:size(fixed_points, 1)
    x_val = fixed_points(i, 1);
    y_val = fixed_points(i, 2);
    plot3([x_val, x_val], [y_val, y_val], [0, line_length], 'r--', 'LineWidth', 2, 'DisplayName', 'Real Target Values');
end

% Add legend
legend({'Cost Function Values', 'Real Target Values'}, 'Location', 'best', 'FontSize', 20);

hold off;
