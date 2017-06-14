function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

multiplyer=ones(size(theta));
multiplyer(1)=0;

h=X*theta;

delta=h-y;

Junreg=1/(2*m)*(delta'*delta);

J=Junreg+lambda/(2*m)*((theta.^2)'*multiplyer);

gradUnreg=(delta'*X./m)';

grad=gradUnreg+lambda/m*(theta.*multiplyer);


% =========================================================================

grad = grad(:);

end
