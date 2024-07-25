function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha
% Initialize some useful values
data = load('ex1data2.txt'); %loading data set 
[m,n]=size(data); %getiing dimensions of data.txt
y = data(:, n); %definging o/p varibles from data set
X = [ones(m, 1), data(:,1:n-1)]; % Add a column of ones to x
%m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
   
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
   for i = 1:size(X,2)
       for j = 1:length(theta)
         error= X(i,j)*theta(j) - y(i); %calculate error
       end
   end
        
            for j = 1:length(theta)
                for i = 1:size(X,2)
                theta(j) = theta(j) - (alpha*error'*X(i,j))/m;
            end
        end


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
