function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
data = load('ex1data2.txt'); %loading data set 
[m,n]=size(data); %getiing dimensions of data.txt
y = data(:, n); %definging o/p varibles from data set
X = [ones(m, 1), data(:,1:n-1)]; % Add a column of ones to x


% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
error= X*theta - y; %calculate error
Sq_error = (error).^2; %calculate sq error
J=J+(sum(Sq_error))/(2*m);



% =========================================================================

end
