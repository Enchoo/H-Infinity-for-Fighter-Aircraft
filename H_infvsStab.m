yalmip('clear')

% Settings for SeDuMi: precision and maximum iterations
ops_sedumi = sdpsettings('solver', 'sedumi', 'sedumi.eps', 1e-12, 'sdpa.maxIteration', 100);
ops_mosek = sdpsettings('solver', 'mosek', 'sedumi.eps', 1e-12, 'sdpa.maxIteration', 100);

% Fighter aircraft system
A = [-0.0226  -36.6170  -18.8970  -32.0900  3.2509    -0.7626;
     0.0001   -1.8997   0.9831    -0.0007   -0.1708   -0.0050;
     0.0123   11.7200   -2.6316   0.0009    -31.6040  22.3960;
     0        0         1.0000    0         0         0;
     0        0         0         0         -30.0000  0;
     0        0         0         0         0         -30.0000];
B = [0 0; 0 0; 0 0; 0 0; 30 0; 0 30];
C = [0 1 0 0 0 0; 0 0 0 1 0 0];
D = zeros(2);
E = [0 0; 0 0; 0 0; 0 0; 30 0; 0 30];
F = zeros(2);

evA = eig(A); % 2 unstable eigenvalues

Ts = 0.01; % sample time in seconds
T = 25; % sample size

epsilon_ineq = 10.^(-8); % used for strict inequalities
epsilon_noise = 0.1; % noise bound

% Matrix dimensions
n = size(A, 1); % A is nxn
m = size(B, 2); % B is nxm
p = size(C, 1); % C is pxn, D is pxm
d = size(E, 2); % E is nxd, F is pxd

% Discretising system
sysc = ss(A, B, C, D); % continuous-time system
sysd = c2d(sysc, Ts); % discretised system

% Individual noise sample bounds + within a subspace
W_minus = 0.1.*rand(d, T); % norm^2(w) <= epsilon_noise

Phi_11_hat = W_minus * W_minus'; % Best possible bound on the noise
Phi_12_hat = zeros(d, T);
Phi_21_hat = Phi_12_hat';
Phi_22_hat = -eye(T);
Phi_hat = [Phi_11_hat Phi_12_hat; Phi_21_hat Phi_22_hat]; % d+T x d+T
Phi = [E zeros(n, T); zeros(T, d) eye(T)] * Phi_hat * [E zeros(n, T); zeros(T, d) eye(T)]';

% Generating data
x_initial = randn(n, 1);
U_minus = 20.*randn(m, T);
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


% ---Start H-infinity--- %

% Checking observability
OB = obsv(sysd.A, sysd.C);
unobsv = length(A) - rank(OB);
if unobsv ~= 0
    disp('The system is not observable.')
end

% Calculating N
N_leftmult_Phi_postschur = [eye(n) X_plus; zeros(p, n) zeros(p, T); zeros(n, n) -X_minus; zeros(m, n) -U_minus; zeros(n, n) zeros(n, T)];
N = N_leftmult_Phi_postschur * Phi * N_leftmult_Phi_postschur';

% S-lemma - checking if N has at least 1 positive eigenvalue
if all(eig(N) < 0)
    disp('S-lemma is not satisfied.')
end

% Creating variables for the LMI
rho = sdpvar(1);
alpha = sdpvar(1); % >= 0
Q = sdpvar(n, n); % > 0
L = sdpvar(m, n, 'full'); % L = KQ; K = L * inv(Q)

% Supply rate
% F_hat = -1/(gamma.^2).* eye(d), gamma > 0
F_hat = rho.*eye(d); % rho = -1/(gamma.^2) < 0, then gamma = sqrt(-1/rho)
G_hat = zeros(d, p);
H_hat = eye(p);

% Calculating M
M_11 = [Q+E*F_hat*(E')               -E*G_hat+E*F_hat*(F')                                                    zeros(n, n)  zeros(n, m);
        -(G_hat')*(E')+F*F_hat*(E')  -C*Q*(C')-C*(L')*(D')-D*L*(C')+H_hat-(G_hat')*(F')-F*G_hat+F*F_hat*(F')  -C*Q         -C*(L');
        zeros(n, n)                  -Q*(C')                                                                  zeros(n, n)  zeros(n, m);
        zeros(m, n)                  -L*(C')                                                                  zeros(m, n)  zeros(m, m)];
M_12 = [zeros(n, n); D*L; Q; L];
M_21 = [zeros(n, n) (L')*(D') Q L'];
M_22 = Q;
M = [M_11 M_12; M_21 M_22];

LMI = M - alpha.* N; % M and N are post-schur complement argument

conditions_h_inf = [Q >= epsilon_ineq*eye(n), -rho >= epsilon_ineq, alpha >= 0, LMI >= 0];

% ---End H-infinity--- %


% ---Start Stabilisation--- %

% Creating variables for the LMI
beta = sdpvar(1); % > 0
P = sdpvar(n, n); % > 0
L_stab = sdpvar(m, n, 'full'); % L_stab = K_stab * P; K_stab = L_stab * inv(P)

% Calculating N
N_leftmult_Phi_postschur_stab = [eye(n) X_plus; zeros(n, n) -X_minus; zeros(m, n) -U_minus; zeros(n, n) zeros(n, T)];
N_stab = N_leftmult_Phi_postschur_stab * Phi * N_leftmult_Phi_postschur_stab';

% Calculating M
M_stab = [P-beta.*eye(n)  zeros(n, n)   zeros(n, m)  zeros(n, n);
          zeros(n, n)     -P            -(L_stab')   zeros(n, n);
          zeros(m, n)     -L_stab       zeros(m, m)  L_stab;
          zeros(n, n)     zeros(n, n)   L_stab'      P];

LMI_stab = M_stab - N_stab; % M and N are post-schur complement argument

conditions_stab = [P >= epsilon_ineq*eye(n), beta >= epsilon_ineq, LMI_stab >= 0];

% ---End Stabilisation--- %


conditions = [conditions_h_inf, conditions_stab];
% optimize(conditions, rho, ops_sedumi); % no feasible solutions
optimize(conditions, rho, ops_mosek);

% Outputting values from H-infinity
rho = value(rho); % rho has to be negative, otherwise gamma is complex
if rho > 0
    disp('Increase sample size.')
end
gamma = sqrt(-1/rho);
Q = value(Q);
L = value(L);
K = L * inv(Q); %#ok<MINV> % Controller

% Outputting values from Stabilisation
P = value(P);
L_stab = value(L_stab);
K_stab = L_stab * inv(P); %#ok<MINV> % Controller

% Checking stability
sys_new_h_inf = ss(sysd.A+sysd.B*K, [], sysd.C+sysd.D*K, [], -1);
sys_new_stab = ss(sysd.A+sysd.B*K_stab, [], sysd.C+sysd.D*K_stab, [], -1);
disp('The H-infinity controller is stabilising?:')
isstable(sys_new_h_inf)
disp('The stabilisation controller is stabilising?:')
isstable(sys_new_stab)

% Finding gamma for the true system
R = sdpvar(n, n); % >= 0
rho_true = sdpvar(1); % > 0; then gamma_true = sqrt(rho_true)

dissineq = [eye(n) zeros(n, d); sysd.A+sysd.B*K E]' * [R zeros(n, n); zeros(n, n) -R] * [eye(n) zeros(n, d); sysd.A+sysd.B*K E] + [zeros(d, n) eye(d); sysd.C+sysd.D*K F]' * [rho_true*eye(d) zeros(d, p); zeros(p, d) -eye(p)] * [zeros(d, n) eye(d); sysd.C+sysd.D*K F];
conditions_true = [R >= 0, dissineq >= 0, rho_true >= epsilon_ineq];

optimize(conditions_true, rho_true, ops_mosek);
rho_true = value(rho_true);
gamma_true = sqrt(rho_true);

% Generating outputs for random noise
W_rand = 0.1.*rand(d, T);
X_hinf_rand = [x_initial zeros(n, T)];
Y_hinf_rand = zeros(p, T);
for i = 1:T
    x_next_hinf_rand = (sysd.A + sysd.B * K)*X_hinf_rand(:, i) + E*W_rand(:, i);
    X_hinf_rand(:, i+1) = x_next_hinf_rand;
    y_next_hinf_rand = (sysd.C + sysd.D * K)*X_hinf_rand(:, i);
    Y_hinf_rand(:, i) = y_next_hinf_rand;
end

X_stab_rand = [x_initial zeros(n, T)];
Y_stab_rand = zeros(p, T);
for i = 1:T
    x_next_stab_rand = (sysd.A + sysd.B * K_stab)*X_stab_rand(:, i) + E*W_rand(:, i);
    X_stab_rand(:, i+1) = x_next_stab_rand;
    y_next_stab_rand = (sysd.C + sysd.D * K_stab)*X_stab_rand(:, i);
    Y_stab_rand(:, i) = y_next_stab_rand;
end

% Generating outputs for sin/cos noise
X_hinf_sc = [x_initial zeros(n, T)];
Y_hinf_sc = zeros(p, T);
for i = 1:T
    x_next_hinf_sc = (sysd.A + sysd.B * K)*X_hinf_sc(:, i) + E*[sin(i); cos(i)];
    X_hinf_sc(:, i+1) = x_next_hinf_sc;
    y_next_hinf_sc = (sysd.C + sysd.D * K)*X_hinf_sc(:, i);
    Y_hinf_sc(:, i) = y_next_hinf_sc;
end

X_stab_sc = [x_initial zeros(n, T)];
Y_stab_sc = zeros(p, T);
for i = 1:T
    x_next_stab_sc = (sysd.A + sysd.B * K_stab)*X_stab_sc(:, i) + E*[sin(i); cos(i)];
    X_stab_sc(:, i+1) = x_next_stab_sc;
    y_next_stab_sc = (sysd.C + sysd.D * K_stab)*X_stab_sc(:, i);
    Y_stab_sc(:, i) = y_next_stab_sc;
end

% Plotting the results
x_axis = 0:1:(T-1);

figure(1) % angle of attack and attitude angle, random noise
tiledlayout(2,1)
nexttile
plot(x_axis, Y_hinf_rand(1, :))
hold on
plot(x_axis, Y_stab_rand(1, :))
hold off
xlabel('Time (sec)') 
ylabel('Angle of Attack')
legend('H-infinity','Stabilising')
nexttile
plot(x_axis, Y_hinf_rand(2, :))
hold on
plot(x_axis, Y_stab_rand(2, :))
hold off
xlabel('Time (sec)') 
ylabel('Attitude Angle')
legend('H-infinity','Stabilising')
sgtitle('H-infinity vs Stabilising Controller with Random Noise')

figure(2) % angle of attack and attitude angle, sin/cos noise
tiledlayout(2,1)
nexttile
plot(x_axis, Y_hinf_sc(1, :))
hold on
plot(x_axis, Y_stab_sc(1, :))
hold off
xlabel('Time (sec)') 
ylabel('Angle of Attack')
legend('H-infinity','Stabilising')
nexttile
plot(x_axis, Y_hinf_sc(2, :))
hold on
plot(x_axis, Y_stab_sc(2, :))
hold off
xlabel('Time (sec)') 
ylabel('Attitude Angle')
legend('H-infinity','Stabilising')
sgtitle('H-infinity vs Stabilising Controller with sin/cos Noise')
