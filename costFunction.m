function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

xTheta = X*theta;
sigmoidForOne = sigmoid(-xTheta);

hMinusY = sigmoidForOne - y;

grad = transpose(X)*hMinusY;
grad = grad/m;

% cost function code

sigmoidForZero = 1 - sigmoidForOne;
logSigmoidOne = log(sigmoidForOne);
logSigmoidZero = log(sigmoidForZero);

yForOutputZero = 1 - y;

jForOne = transpose(y)*logSigmoidOne;
jForZero = transpose(yForOutputZero)*logSigmoidZero;

J = jForOne + jForZero;

J = J * (-1);

J = J/m;


% =============================================================

end
