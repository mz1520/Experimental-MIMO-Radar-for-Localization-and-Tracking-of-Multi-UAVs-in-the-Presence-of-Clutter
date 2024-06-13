%Reshapes the data as a bank of TDLs
function [x_n] = threed_cube(x_t, goldseq)
    % Get the size of x_t
    [N, L] = size(x_t);
    Next = length(goldseq) * 2;
    
    % Calculate the final length 
    num_windows = floor(L / Next);
    
  
    x_n = zeros(N * Next, num_windows);
    
    % Loop over each window in x_t to populate x_n
    for win = 1:num_windows
        
        start_idx = (win - 1) * Next + 1;
        end_idx = win * Next;
        
        % Create the column by concatenating Next samples from each row of x_t
        for row = 1:N
            x_n((row-1)*Next+1 : row*Next, win) = x_t(row, start_idx:end_idx);
        end
    end
end




