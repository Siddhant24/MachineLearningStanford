function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
        
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Z2 = [ones(m, 1) X] * Theta1';
A2 = sigmoid(Z2);
Z3 = [ones(m, 1) A2] * Theta2';
h = sigmoid(Z3);

for i = 1:m
    yVec = zeros(num_labels, 1);
    yVec(y(i,1), 1) = 1;
    J = J + log(h(i, :)) * yVec + log(ones(1,num_labels) - h(i, :)) * (ones(num_labels, 1) - yVec);
end

J = (-1/m) * J;

for i = 1:hidden_layer_size
    for k = 2:input_layer_size + 1
        J = J + (lambda/(2*m)) * Theta1(i,k)^2;
    end
end

for i = 1:num_labels
    for k = 2:hidden_layer_size + 1
        J = J + (lambda/(2*m)) * Theta2(i,k)^2;
    end
end

Del1 = zeros(size(Theta1));
Del2 = zeros(size(Theta2));

for t = 1:m
    a1 = X(t,:)';
    a1 = [1;a1];
    z2 = Z2(t, :)';
    a2 = A2(t, :)';
    a2 = [1;a2];
    z3 = Z3(t, :)';
    a3 = h(t, :)';
    yVec = zeros(num_labels, 1);
    yVec(y(t,1), 1) = 1;
    delta3 = a3 - yVec; 
    delta2 = Theta2' * delta3 .* sigmoidGradient([1;z2]);
    delta2 = delta2(2:end);
    Del2 = Del2 + delta3 * a2';
    Del1 = Del1 + delta2 * a1';
end;
Theta2_grad = (1/m) * Del2;
Theta1_grad = (1/m) * Del1;

for i = 1:size(Theta2, 1)
    for j = 2:size(Theta2, 2)
        Theta2_grad(i,j) = Theta2_grad(i,j) + (lambda/m) * Theta2(i,j);
    end
end
for i = 1:size(Theta1, 1)
    for j = 2:size(Theta1, 2)
        Theta1_grad(i,j) = Theta1_grad(i,j) + (lambda/m) * Theta1(i,j);
    end
end

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
