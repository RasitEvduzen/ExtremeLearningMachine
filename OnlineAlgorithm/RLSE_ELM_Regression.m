%% Recursive Least Squares Based Extrame Learning Machine - Nonlinear Regression
% Written By: Rasit
% 09-Mar-2024
clc,clear all,close all;
%% Input and Output definition
xtrain = [1:0.1:20]';
NoD = length(xtrain);
ytrain = 0.01*xtrain.*xtrain + 0.1*exp(-xtrain) + sin(xtrain) +0.1*randn(NoD,1);
NuberOfNeuron = 50;  % Number of Neuron   (Sparse Model)
% NuberOfNeuron = NoD;  % Number of Neuron

%% Select Activation Function
input_w = randn(NuberOfNeuron,1);  % generate a random input weights
H = radbas(xtrain*input_w');   % Radial Basis Activation Function
% H = tanh(xtrain*input_w');     % TanH Activation Function
% H = cos(xtrain*input_w');      % Cos basis Function
% H = sin(xtrain*input_w');      % Sin basis Function
% H = 1 ./ (1 + exp(-xtrain*input_w'));  % Sigmoid Function

%% RLSE Solution
% output_w = pinv(H)*ytrain; % LSE solution

A = H;
b = ytrain;
output_w = rand(size(A,2),1);     % Random start RLSE state vector
P = 1e10*eye(size(A,2),size(A,2)); % This tunable parameter depend on activation function

figure('units','normalized','outerposition',[0 0 1 1],'color','w')
for k=1:NoD
    [output_w,K,P] = rlse_online(A(k,:),b(k,:),output_w,P);
    ElmOutput = (H*output_w); % calculate the actual output
    % Plot Result
    clf
    plot(xtrain, ytrain,'ko-',LineWidth=2),hold on;
    hold on,grid minor
    axis([0 21 -2 5.5])
    plot(xtrain,ElmOutput,'r.-',LineWidth=2),title({"Number of Neurons: "+num2str(NuberOfNeuron)})
    legend('Training Data','RLSE-ELM Output')
    drawnow
end


function [x,K,P] = rlse_online(a_k,b_k,x,P)
% One step of RLSE (Recursive Least Squares Estimation) algorithm
a_k = a_k(:);
b_k = b_k(:);
K = (P*a_k)/(a_k'*P*a_k+1); % Compute Gain K (Like Kalman Gain!)
x = x + K*(b_k-a_k'*x);     % State Update
P = P - K*a_k'*P;           % Covariance Update
end
