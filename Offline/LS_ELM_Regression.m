%% Recursive Least Squares Based Extrame Learning Machine - Nonlinear Classification
% Written By: Rasit
% 09-Mar-2024
clc,clear all,close all;
%% Input and Output definition
xtrain = [1:0.1:20]';
NoD = length(xtrain);
ytrain = 0.01*xtrain.*xtrain + 0.1*exp(-xtrain) + sin(xtrain) +0.1*randn(NoD,1);
NuberOfNeuron = 50;  % Number of Neuron   (Sparse Model)
% NuberOfNeuron = NoD;  % Number of Neuron

input_w = randn(NuberOfNeuron,1);  % generate a random input weights
H = radbas(xtrain*input_w');   % Radial Basis Activation Function 
% H = tanh(xtrain*input_w');     % TanH Activation Function
% H = cos(xtrain*input_w');      % Cos basis Function
% H = sin(xtrain*input_w');      % Sin basis Function
% H = 1 ./ (1 + exp(-xtrain*input_w'));  % Sigmoid Function
output_w = pinv(H)*ytrain; % (LSE solve) 
ElmOutput = (H*output_w); % calculate the actual output

% Plot Result
figure('units','normalized','outerposition',[0 0 1 1],'color','w')
plot(xtrain, ytrain,'ko-',LineWidth=2),hold on;
hold on,grid minor
plot(xtrain,ElmOutput,'r.-',LineWidth=2),title('LS-ELM Nonlinear Regression')
legend('Training Data','ELM Output')

