yalmip('clear')
A = [-0.0226 -36.6170 -18.8970 -32.0900 3.2509 -0.7626;
     0.0001 -1.8997 0.9831 -0.0007 -0.1708 -0.0050;
     0.0123 11.7200 -2.6316 0.0009 -31.6040 22.3960;
     0 0 1.0000 0 0 0;
     0 0 0 0 -30.0000 0;
     0 0 0 0 0 -30.0000];
B = [0 0; 0 0; 0 0; 0 0; 30 0; 0 30];
C = [0 1 0 0 0 0; 0 0 0 1 0 0];
D = zeros(2);
E = [0 0; 0 0; 0 0; 0 0; 30 0; 0 30];
F = zeros(2);

Ts = 0.01; % sampling time
%T = 100; % T = sample size

epsilon_ineq = 10.^(-10); % for strict inequalities
epsilon_noise = 1; % noise bound

evA = eig(A); % 2 unstable eigenvalues

sample_sizes = 60:20:200;
y_axis = zeros(1, length(sample_sizes));
for i = 1:length(sample_sizes)
    T = sample_sizes(1, i);
    [gamma, K, unobsv] = H_inf(Ts, T, A, B, C, D, E, F, epsilon_ineq, epsilon_noise);
    y_axis(1, i) = gamma;
end

plot(sample_sizes, y_axis)

function [gamma, K, unobsv] = H_inf(Ts, T, A, B, C, D, E, F, epsilon_ineq, epsilon_noise)
% matrix dimensions
n = size(A, 1); % A is nxn
m = size(B, 2); % B is nxm
p = size(C, 1); % C is pxn
d = size(E, 2); % E is nxd

sysc = ss(A, B, C, D); % continuous-time system
sysd = c2d(sysc, Ts); % discretised system
% checking observability
OB = obsv(sysd.A, sysd.C);
unobsv = length(A) - rank(OB);

if unobsv ~= 0
    disp('The system is not observable.')
   
else
    % individual noise sample bounds + within a subspace
    Phi_11_hat = epsilon_noise.*T.*eye(d);
    Phi_12_hat = zeros(d, T);
    Phi_21_hat = Phi_12_hat.';
    Phi_22_hat = -eye(T);
    Phi_hat = [Phi_11_hat Phi_12_hat; Phi_21_hat Phi_22_hat]; % d+T x d+T
    Phi = [E zeros(n, T); zeros(T, d) eye(T)] * Phi_hat * [E zeros(n, T); zeros(T, d) eye(T)].';
    W_minus = 0.6.*rand(d, T); % norm^2(w) <= epsilon_noise
    
    x_initial = randn(n, 1);
    U_minus = 3.*randn(m, T);
    X = [x_initial zeros(n, T)];
    Y_minus = zeros(p, T);
    
    for i = 1:T
        x_next = sysd.A*X(:, i) + sysd.B*U_minus(:, i) + E*W_minus(:, i);
        X(:, i+1) = x_next;
        y_next = sysd.C*X(:, i) + sysd.D*U_minus(:, i);
        Y_minus(:, i) = y_next;
    end
    X_minus = X(:, 1:end-1);
    X_plus = X(:, 2:end);
    
    N_leftmult_Phi_postschur = [eye(n) X_plus; zeros(p, n) zeros(p, T); zeros(n, n) -X_minus; zeros(m, n) -U_minus; zeros(n, n) zeros(n, T); zeros(n, n) zeros(n, T)];
    N = N_leftmult_Phi_postschur * Phi * transpose(N_leftmult_Phi_postschur);

    if any(eig(N) > 0)
        rho = sdpvar(1);
        alpha = sdpvar(1); % >= 0
        Q = sdpvar(n, n); % > 0
        L = sdpvar(m, n, 'full'); % L = KQ; K = L * inv(Q)
        
        % Storage function
        % F_hat = -1/(gamma.^2).* eye(d), gamma > 0
        F_hat = rho.* eye(d); % rho = -1/(gamma.^2) < 0, then gamma = sqrt(-1/rho)
        G_hat = zeros(d, p);
        H_hat = eye(p);
        
        M_11 = [Q+E*F_hat*(E.') -E*G_hat+E*F_hat*(F.') zeros(n, n);
                -(G_hat.')*(E.')+F*F_hat*(E.') -C*Q*(C.')-C*(L.')*(D.')-D*L*(C.')+H_hat-(G_hat.')*(F.')-F*G_hat+F*F_hat*(F.') D*L;
                zeros(n, n) (L.')*(D.') Q];
        M_12 = [zeros(n, n) zeros(n, m) zeros(n, n); -C*Q -C*(L.') D*L; Q L.' Q];
        M_21 = [zeros(n, n) -Q*(C.') Q; zeros(m, n) -L*(C.') L; zeros(n, n) (L.')*(D.') Q];
        M_22 = [zeros(n, n) zeros(n, m) Q; zeros(m, n) zeros(m, m) L; Q L.' Q];
        M = [M_11 M_12; M_21 M_22];
        
        LMI = M - alpha.* N; % M and N are post-schur complement arguments
        
        conditions = [Q >= epsilon_ineq.*eye(n), -rho >= epsilon_ineq, alpha >= 0, LMI >= 0];
        
        optimize(conditions, rho);
        rho = value(rho);
        gamma = sqrt(-1/rho);
        Q = value(Q);
        L = value(L);
        K = L * pinv(Q);
        
        if rho > 0
            disp('Increase sample size.')
        end
    else
        disp('S-lemma not satisfied.')
    end
end
end
