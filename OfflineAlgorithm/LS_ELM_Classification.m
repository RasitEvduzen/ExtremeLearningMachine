%% Recursive Least Squares Based Extrame Learning Machine - Nonlinear Regression
% Written By: Rasit
% 09-Mar-2024
clc,clear all,close all;
%% Input and Output definition
% Create Data Spiral Data
B = 4;
N = 200;
Tall = [];
for i=1:N/2
    theta = pi/2 + (i-1)*[(2*B-1)/N]*pi;
    Tall = [Tall , [theta*cos(theta);theta*sin(theta)]];
end
Tall = [Tall,-Tall];
Tmax = pi/2+[(N/2-1)*(2*B-1)/N]*pi;
xtrain = [Tall]'/Tmax;
ytrain = [-ones(1,N/2), ones(1,N/2)]';

NoD = length(xtrain);
% NuberOfNeuron = 50;  % Number of Neuron   (Sparse Model)
NuberOfNeuron = NoD;  % Number of Neuron

input_w = 2*randn(NuberOfNeuron,size(xtrain,2));  % generate a random input weights
H = tanh(xtrain*input_w');     % TanH Activation Function
output_w = pinv(H)*ytrain; % (LSE solve) 
ElmOutput = (H*output_w); % calculate the actual output

%% Model Test
figure('units','normalized','color','w')

xtest = []; 
ytest = [];

for t1=-1:1e-2:1
    for t2=-1:1e-2:1
        xtest = [xtest; [t1,t2]];
    end
end

Ht = tanh(xtest*input_w');
ytest = sign((Ht*output_w));

plot(xtest(ytest==+1,1),xtest(ytest==+1,2),'r.'),hold on,grid on
plot(xtest(ytest==-1,1),xtest(ytest==-1,2),'b.')

plot(xtrain(ytrain==+1,1), xtrain(ytrain==+1,2), 'k*',LineWidth=5)
plot(xtrain(ytrain==-1,1), xtrain(ytrain==-1,2), 'y*',LineWidth=5)
title('LS-ELM Nonlinear Classification')

