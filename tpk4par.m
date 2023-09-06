function [r,kappa,tau] = tpk4par(x,par_f) 
%
% r is a weighted sum of tau and kappa 
% depending on the 4 quantities
% alpha                             in [0,2]  (power in step length)
% beta                              in [0,5]  (weight w_k = k^beta)
% M = 1 + delta1 * kmax with delta1 in [0,1]  (offset)
% c = delta2 * (1/Dnn)  with delta2 in [0,1]  (step length reduction)
%
% Compute EXACT VALUES Gik(i+1) = ||G_{i,k}|| for 0 \le i \le k-1
% (do ||G_{k-1,k}|| separately)

alpha  = x(1);  % must be in [0,2]  
beta   = x(2);  % must be in [0,5] 
delta1 = x(3);  % must be in [0,1]  
delta2 = x(4);  % must be in [0.1,1]

D11  = par_f.D11; 
Dnn  = par_f.Dnn; 
k    = par_f.k; 
eta  = par_f.eta; 
c    = delta2/Dnn; 
barc = D11*c; 


if D11 <= 0 || Dnn < D11 || eta < 0 || k > 1e8 || k < 3
    error('unreasonable problem parameters');
end
    

M    = 1+delta1*k;
tmp1 = 0:k+1; 
tmp2 = (M./(M+tmp1)).^alpha; % tmp2(i) = (M/(M+i-1))^alpha

C0j    = ones(k+1,1);        % C0j(j) = C_{0,j-1}  
C0j(1) = 1-barc;


Gik    = zeros(k,1);                    % Gik(i+1) = ||G_{i,k}|| 
Gik(k) = c*k^beta*(M/(k-1+M))^alpha;    % Gik(k  ) = ||G_{k-1,k}||
tau    = 0; % (sum j^beta)^(-1) sum j^beta ||C_{0,j-1}||  (1\le j\le k)
kappa  = 0; % (sum j^beta)^(-2) sum ||G_{i,k}||^2         (0\le i\le k-1)
sjbeta = 0; % sum j^beta


for j = 1:k-1
    C0j(j+1) = C0j(j)*(1-barc*tmp2(j+1));
    Gik(k-j) = c*(k-j)^beta*tmp2(k-j) + (1-barc*tmp2(k-j+1))*Gik(k-j+1)...
               *((k-j+M)/(k-j-1+M))^alpha;
    ibeta    = j^beta;
    sjbeta   = sjbeta + ibeta;
    tau      = tau + ibeta*C0j(j);
    kappa    = kappa + Gik(k-j)^2;
end

ibeta  = k^beta;
sjbeta = sjbeta + ibeta;
tau    = tau + ibeta*C0j(k);
kappa  = kappa + Gik(k)^2;

tau   = tau/sjbeta;
kappa = kappa / sjbeta^2;
kappa = sqrt(kappa);

r = tau + eta*kappa; 
r = r/(1+eta);  

end
