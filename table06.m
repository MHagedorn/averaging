% supplementary material on "Optimized convergence of stochastic gradient
% descent by weighted averaging" (2022)

clear all
clc


% a posteriori validation of the parameter recommendation
% alpha = 0, beta = 2.4, delta = 0, c = 1


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% unweighted averaging (Polyak and Juditsky)
par0 = [0,   0, 0, 1];

% weighted averaging (recommended parameter selection)
par  = [0, 2.4, 0, 1];


eta  = 0; % dummy, not used
D11s = 10.^(-0.5:-0.5:-4);
Dnn  = 1;
kmax = 10^4;
% optimal values for tau due to table 5:
taus = [9e-12, 4e-10, 1e-9, 6e-9, 1e-5, 2e-3, 0.07, 0.4]; 

results = zeros(7, length(D11s));

for k = 1:length(D11s)
    D11 = D11s(k);

options.par_fobj.D11 = D11; 
options.par_fobj.Dnn = Dnn; 
options.par_fobj.k   = kmax; 
options.par_fobj.eta = eta; 
[~, kappa0, tau0]    = tpk4par(par0, options.par_fobj); % Juditsky
[~,  kappa,  tau]    = tpk4par( par, options.par_fobj); % recommended
errstoch = -(kappa0 - kappa)/kappa0;

results(1,k) = 1/D11;
results(2,k) = 1/(kmax*D11);
results(3,k) = kappa0;
results(4,k) = errstoch;
results(5,k) = tau0;
results(6,k) = taus(k);
results(7,k) = tau;

end


disp('Table 6:')
T = table(results(:,1),results(:,2),results(:,3),results(:,4),...
    results(:,5),results(:,6),results(:,7),results(:,8));
T.Properties.RowNames = {'cond(D)','cond/kmax','kappa(1,1)',...
    'kappa(w,gamma)','tau(1,1)','tau(w*,gamma*)','tau(w,gamma)'};
T