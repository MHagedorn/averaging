% supplementary material on "Optimized convergence of stochastic gradient
% descent by weighted averaging" (2022)

clear all
clc

% solve
% min tau s.t. kappa \leq sqrt(2)*kappa(1,1), alpha in [0,2], 
%              beta in [0,5], delta in [0,1], c in [0.1,1]

% or equivalently
% min v   s.t. kappa \leq sqrt(2)*kappa(1,1), alpha in [0,2], 
%              beta in [0,5], delta in [0,1], c in [0.1,1], tau \leq v

% find optimal parameters for different values of cond(D)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% x = (alpha, beta, delta, c)
eta     = 0.017; % dummy, not used
D11s    = 10.^(-0.5:-0.5:-4);
Dnn     = 1;
kmax    = 10^4;
results = zeros(10, length(D11s));

for k = 1:length(D11s)
    D11 = D11s(k);

options.par_fobj.D11 = D11; 
options.par_fobj.Dnn = Dnn; 
options.par_fobj.k   = kmax; 
options.par_fobj.eta = eta; 
[~, kappa0, tau0]    = tpk4par([0, 0, 0, 1], options.par_fobj); % Juditsky

options.lb = [    0; 0; 0; 0; 0.1]; % lower bounds
options.ub = [ tau0; 2; 5; 1;   1]; % upper bounds

options.par_fcon.D11    = D11; 
options.par_fcon.Dnn    = Dnn; 
options.par_fcon.k      = kmax; 
options.par_fcon.eta    = eta; 
options.par_fcon.kappa0 = kappa0;



rng(1234)
times = 10;
xopts = zeros(length(options.lb), times);
taus  = zeros(1, times);
for i = 1:times
pars = rand(1, 4);
x0 = [tau0; pars(1)*2; pars(2)*5; pars(3); pars(4)*0.9 + 0.1];
options.x0 = x0;

[x, ~, ~, out]  = min_fc(@f_obj, @f_con, options);
xopts(:, i) = x;

[~, ~, tau] = tpk4par(x(2:end), options.par_fobj);
taus(:, i)  = tau;
end
%mean(xopts,2);         % for additional analysis
%sqrt(var(xopts, 0, 2))
[~, argmin] = min(taus);
x = xopts(:, argmin);

[~, kappa, tau] = tpk4par(x(2:end), options.par_fcon);
errstoch = -(kappa0 - kappa)/kappa0;

results(1,k)  = 1/D11;
results(2,k)  = 1/(kmax*D11);
results(3,k)  = kappa0;
results(4,k)  = errstoch;
results(5,k)  = tau0;
results(6,k)  = tau;
results(7,k)  = x(2);
results(8,k)  = x(3);
results(9,k)  = x(4);
results(10,k) = x(5);

end


disp('Table 5:')
T = table(results(:,1),results(:,2),results(:,3),results(:,4),...
    results(:,5),results(:,6),results(:,7),results(:,8));
T.Properties.RowNames = {'cond(D)','cond/kmax','kappa(1,1)',...
    'kappa(w*,gamma*)','tau(1,1)','tau(w*,gamma*)','alpha*','beta*',...
    'delta*','c*'};
T


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function definitions follow below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [y] = f_obj(x, opt)

y = x(1);

end


function [f_e, f_i] = f_con(x, opt)

kappa0 = opt.kappa0;
[~,  kappa, tau] = tpk4par(x(2:end), opt);

f_e = [];
f_i = zeros(2,1);
f_i(1) = kappa - sqrt(2)*kappa0;
f_i(2) = tau - x(1);

end
