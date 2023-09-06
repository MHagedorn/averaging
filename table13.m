% supplementary material on "Optimized convergence of stochastic gradient
% descent by weighted averaging" (2022)

% Test environment for Algorithm 1 by Hagedorn and Jarre
% for the logistic reformulation of the MNIST example,
% http://yann.lecun.com/exdb/mnist/
clear all 
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dim  = 60000; % dim is the number m in the paper (must be <= 60000) 
doNewton = 0; % possiby add some Newton steps to improve the approximate
              % solution found by sgd. (doNewton either 0 or 1)

% Read training data and training labels
[imgs, labels] = readMNIST('train-images-idx3-ubyte', ...
    'train-labels-idx1-ubyte', dim, 0);
% imgs is an array of size p*p*dim with values in [0,1]
% labels is a vector of size dim with entries in {0, ..., 9}
disp('training data read in')


[p, ~, ~] = size(imgs); % images with p^2 pixels each
pixel = p^2;            % pixel is the number n in the paper,
                        % 28x28 pixels per image

% For training

sigma = @(x) 1./(1+exp(-x)); % logistic function

z  = 0;            % digit of interest
b1 = labels==z;    % produces a vector with entries 0 and 1,
                   % the i-th entry is 1, if the image is labeled as 0
                   % otherwise the i-th entry is 0
b = b1*2 - 1;      % label +1 or -1 (instead of 1 or 0)
A = reshape(imgs(:,:,:), pixel,dim); % one image per column


Psi_SGD = @(x,S) Psi_partialgrad(x, S, b, A, sigma);
                   % "partial" gradient meaning gradient for the batch S

c = 16/pixel;      % rather short step length for sgd

% For testing
Total = 10000;     % 10000 test images possible
[imgs2, labels2] = readMNIST('t10k-images-idx3-ubyte',...
    't10k-labels-idx1-ubyte', Total, 0);
B = reshape(imgs2(:,:,:), pixel,Total);
disp('test data read in')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% alg1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
kmaxs  = 10.^(3:7);
paropt = [false, true];
sols  = zeros(2*length(paropt), length(kmaxs));

for i = 1:length(kmaxs)
disp(['Step ',num2str(i),'/',num2str(length(kmaxs))])
for j = 1:length(paropt)
    
rng(100922); % seed for reproducibility
kmax  = kmaxs(i);
if paropt(j) == false
    beta  = 0;
    alpha = 0;
    M     = 1;
end
if paropt(j) == true
    beta  = 2.4;
    alpha = 0;
    delta = 0;
    M     = 1 + delta * kmax;
end

m  = dim; % number of summands/training images
nu = 10;  % batch size

x = zeros(pixel,1); % intitial value

sigm  = 0;                     % sum of weights 
                               % ("sigma" is used for the sigmoid function)
xbar  = x;

%%% Now the actual Algorithm 1:
for k = 0:(kmax-1)
    Sk    = randi(m, 1, nu);   % batch S_k, drawing WITH replacement
    w     = (k+1)^beta;        % weight w(k)
    gamma = c * (M/(k + M))^alpha;
    sigm  = sigm + w;          % sum of weights
    grad  = Psi_SGD(x, Sk);
    x     = x - gamma*grad;    % update
    xbar  = xbar + w*x;
end

xbar = xbar/sigm;
%norm(xbar - xopt)
%%% End of the actual Algorithm 1.

s     = sigma(b.*((xbar'*A).'));
dpsi  = (- A*(b.*(1 - s)))/length(b);   % the (full) final gradient
ndpsi = norm(dpsi);                     % norm of the final gradient
disp(['Norm of the Gradient: ',num2str(ndpsi)])

% compare prediction with the test results:
pred = sigma(xbar'*B)';   
pred = pred>0.5;   % if pred(i) == 1, then column i of the test Matrix B
                   % is classifed to represent a ``0''
fcr = 1-sum(pred==(labels2==z))/length(pred); % false classification rate
disp(['False classification rate: ',num2str(fcr)])

sols(j,i)               = ndpsi;
sols(j+length(paropt),i) = fcr;

end
end


T = table([kmaxs; sols]);
T.Properties.RowNames = {'k^max', 'norm(gradient), w^0, gamma^0',...
    'norm(gradient), w^opt, gamma^opt', 'FCR, w^0, gamma^0',...
    'FCR, w^opt, gamma^opt'};
T.Properties.VariableNames = {'Table 13'}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% further optimization with Newton for comparison
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if doNewton == 1

disp(' ')
disp('Results with additional Newton steps')
Psi_Newton = @(w) Psi_Hesse(w, b, A, pixel, sigma);

x = xbar;
[xopt,steps, nx, ng] = newtonmethod(x,Psi_Newton, b, A);
norm(x-xopt);
s     = sigma(b.*((xopt'*A).'));
dpsi  = ( - A*(b.*(1 - s)))/length(b); 
ndpsi = norm(dpsi);      % norm of the gradient near the optimal point
disp(['Norm of the Gradient: ',num2str(ndpsi)])
pred2 = sigma(xopt'*B)'; % vector of probabilities
pred2 = pred2>0.5;
diff = sum(pred~=pred2); 
fcr = 1-sum(pred2==(labels2==z))/length(pred2); % false classif. rate
disp(['False classification rate: ',num2str(fcr)])
disp(['Norm(xSGD-xopt)',num2str(norm(x-xopt))])
disp(['Number of different decisions: ',num2str(diff)])

loglog(nx,ng)
title('Newton method')
xlabel('norm(x_k)')
ylabel('norm(gradient)')

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function definitions follow below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [grad] = Psi_partialgrad(w, S, b, A, sigma)
% batch gradient of the function Psi = f 
    A    = A(:,S);
    c    = length(S);
    s    = sigma(b(S).*((w'*A).'));
    grad = - A*(b(S).*(1 - s))/c;
end



function [dpsi, ndpsi, ddpsi] = Psi_Hesse(w, b, A, pixel, sigma)
% regularized Hession of the full function Psi = f 
% used for Newton's method only
    gam   = 1e-6; % regularization term added to avoid singular Hessian 
    s2    = sigma(b.*((w'*A).'));
    dpsi  = (- A*(b.*(1 - s2)))/length(b); % gradient
    ndpsi = norm(dpsi);
    B     = A.*sqrt(max(s2.*(1-s2),0).');
    ddpsi = (diag(gam*ones(pixel,1)) + B*B.')/length(b);
end



function [x,steps, nx, ng] = newtonmethod(x, Psi, b, A)
% Newton method with primitive line search for Psi = f of the logistic
% regression example
tol      = 1e-12; % tolerance
stepsize = 1; 
maxstep  = 50;
steps    = 0;
sigma    = @(x) 1./(1+exp(-x));
nx       = zeros(1,maxstep); % for loglog-plot 
ng       = zeros(1,maxstep);
[dpsix, ndpsix, ddpsix] = Psi(x);

while ndpsix > tol && steps < maxstep
    
    steps = steps + 1;
    
    % next 4 lines just for recording...
    s         = sigma(b.*((x'*A).')); 
    dpsi      = (- A*(b.*(1 - s)))/length(b); 
    nx(steps) = norm(x);
    ng(steps) = norm(dpsi); 
    
    deltax    = -ddpsix\dpsix; % Newton update
    
    bisectionsteps = 0;
    ndpsixn        = ndpsix;
    while ndpsixn >= ndpsix && bisectionsteps < 10
       xn              = x + stepsize*deltax; 
       bisectionsteps  = bisectionsteps + 1;
       [~, ndpsixn, ~] = Psi(xn);
       stepsize        = stepsize / 2;
    end
    if bisectionsteps >= 10
        error('bisection failed')
    end
    stepsize = 1;
    x        = xn;
    [dpsix, ndpsix, ddpsix] = Psi(x);
    
end
s           = sigma(b.*((x'*A).'));
dpsi        = (- A*(b.*(1 - s)))/length(b); 
nx(steps+1) = norm(x);
ng(steps+1) = norm(dpsi);
nx          = nx(1:steps+1);
ng          = ng(1:steps+1);
end
