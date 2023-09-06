% supplementary material on "Optimized convergence of stochastic gradient
% descent by weighted averaging" (2022)

% second more realistic case of having different functions f_k
% final error in dependence of the initial error
clear all
clc

rng(12345)                      % to make results reproducible
lambda  = 10;                   % initial value
rhos    = 1*10.^(-1.5:0.5:1.5); % vary noise
rhos    = [0, rhos];
paropt  = [false, true];
sols    = zeros(length(paropt)+1, length(rhos));
Sol     = zeros(length(paropt)+1, length(rhos));

n       = 100;
kmax    = 1e3;
setcond = 10;            % condition of Hessian of f is setcond
setcond = sqrt(setcond); % condition of D = A^{1/2}

for run = 1:10

D = rand(n,1);
D = D-min(D);                    % minimum is zero now
D = D/max(D);                    % in [0,1] now
D = 1/setcond + (1-1/setcond)*D; % entries in [1/setcond, 1]
D = sort(D);                     % here D is as in Section 3.3 
D = D*10;                        % scale D 

c  = 1/max(D)^2;  % (here, D is the square root the matrix D in Section 2)
x0 = ones(n,1);   % initial error "evenly distributed" over all components
x0 = x0/sqrt(n);  % normalize for all dimensions 
x0 = lambda * x0; % initial error

for i = 1:length(rhos)
rho = rhos(i);
   
    
for j = 1:length(paropt)
if paropt(j) == false % averaging
    beta  = 0;
    alpha = 0;
    M     = 1;
end
if paropt(j) == true  % weighted averaging
    beta  = 2.4;
    alpha = 0;
    delta = 0;
    M     = 1 + delta * kmax;
end

x     = x0;          % ``plain'' iterates
xa    = zeros(n,1);  % averages
xm    = x0;          % iterates of momentum method
xp    = x0;          % previous step of momentum method
sigma = 0;

for k = 0:kmax-1
    rk       = 0.1*randn(n,1);
    bk       = rho*randn(n,1);
    ak       = D.*rk;
    nablafk  = ak*(ak.'*x) + bk;
    nablafkm = ak*(ak.'*xm) + bk;
    
    gamma = c * (M/(k + M))^alpha;
    wk    = (k+1)^beta;
    x     = x - gamma*nablafk;
    xa    = xa + wk*x;
    sigma = sigma+wk;
    
    % for momentum method    
    xh = xm;
    xm = xm - 0.001*nablafkm + 0.9*(xm - xp);
    xp = xh;
    
end

xa        = xa/sigma;
xnorm     = norm(x);	    
xanorm    = norm(xa);
xmnorm    = norm(xm);
sols(j,i) = xanorm;
sols(3,i) = xmnorm;
end
end

Sol = Sol + sols;

end

Sol = Sol./10;

disp('Norm of final iterate in dependence of the noise rho:')
T = table(Sol(:,1),Sol(:,2),Sol(:,3),Sol(:,4),Sol(:,5),Sol(:,6),... 
    Sol(:,7), Sol(:,8));
T.Properties.VariableNames = split(num2str(rhos));
T.Properties.RowNames = {'norm(x_average^final), w^0, gamma^0',...
    'norm(x_average^final), w^opt, gamma^opt',...
    'momentum method'}

%%

loglog(rhos(2:end), Sol(1,2:end), rhos(2:end), Sol(2,2:end), '--',... 
    rhos(2:end), Sol(3,2:end), ':', 'LineWidth', 1.5)
legend('averaging (beta = 0)', 'weighted averaging (beta = 2.4)', ...
    'Heavy Ball Method', 'Location', 'NorthWest')
xlabel('noise $\rho$', 'interpreter', 'latex')
ylabel('final error', 'interpreter', 'latex')

%f = gcf;
%exportgraphics(f,'figure1.png','Resolution',300)