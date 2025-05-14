function [tout,x]=pendolo_soluzioneNumerica
% clear; clc;

% Parametri fisici
params.mp = 0.024;     % massa pendolo [kg]
params.mr=0.093;       %massa braccio [kg]
params.Lr = 0.086;     % lunghezza braccio [m]
params.Lp = 0.128;     % lunghezza pendolo [m]
params.Jr = 5.72e-5;    % inerzia braccio [kg m^2]
params.Jp = 1.31e-4;   % inerzia pendolo [kg m^2]
params.Br = 3e-4;    % attrito braccio [Nms]
params.Bp = 5e-4;   % attrito pendolo [Nms]
params.g  = 9.81;    % gravità [m/s^2]
params.Rm= 7.5; %ohm, resistenza interna
params.km=0.0431; %Motor back-emf constant
% Coppia motore (costante o funzione del tempo)
% params.Tm = @(t) 0;  % nessuna coppia applicata

% Condizioni iniziali:
% [gamma0,gammad,phi0,phid]
% [Angolo iniziale cart, velocità angolare iniziale cart, angolo iniziale pendolo, velocità angolare iniziale pendolo]
x0 = [0; 0;  pi/6; 0];

% Intervallo temporale
tspan = linspace(0,15,15000);

% Risoluzione ODE
[t, x] = ode45(@(t,x) pendulum_ODE(t, x, params), tspan, x0);
% Calcolo Vm (non influisce sulla dinamica)
Vm_values = arrayfun(@(ti) Vm_fun(ti), t);

% Output
tout = [t, Vm_values];

% Plot risultati
figure;
plot(t, x(:,1), 'r', 'LineWidth', 1.5); hold on;
plot(t, x(:,3), 'b', 'LineWidth', 1.5); hold on;
plot(t, Vm_values, 'g', 'LineWidth', 1.5);
legend('\gamma (carrello)', '\phi (pendolo)', "Vm")
xlabel('Tempo [s]');
ylabel('Angoli [rad]');
title('Dinamica del pendolo rotante');
grid on;


%% 
function dxdt = pendulum_ODE(t, x, params)
    % Variabili di stato
    gamma   = x(1);  % angolo del cart
    gamma_d = x(2);  % velocità del cart
    phi   = x(3);  % angolo del pendolo
    phi_d = x(4);  % velocità del pendolo

    % Parametri fisici
    mp = params.mp;
    Lr = params.Lr;
    Lp = params.Lp;
    Jr = params.Jr;
    Jp = params.Jp;
    Br = params.Br;
    Bp = params.Bp;
    g  = params.g;
    mr=params.mr;
    Rm=params.Rm;
    km=params.km;
   
 % Calcola Vm in funzione del tempo
    Vm = Vm_fun(t);

    % Coppia motore
    Tm = km * ((Vm - km * gamma_d) / Rm);  

    % Coefficienti intermedi
    C1 = mp * Lr^2 + 0.25 * mp * Lp^2 - 0.25 * mp * Lp^2 * cos(phi)^2 + Jr; %gamma_dd, first equation
    C2 = -0.5 * mp * Lp * Lr * cos(phi); %phi-dd first equation
    C3 = 0.5 * mp * Lp * Lr * cos(phi); %gamma_dd second eq.
    C4 = Jp + 0.25 * mp * Lp^2;

    % Sistema lineare: A * [gamma_dd; phi_dd] = b
    A = [C1, C2;
         C3, C4];

    b = [%first eq.
        Tm - Br * gamma_d ...
        - 0.5 * mp * Lp^2 * sin(phi) * cos(phi) * phi_d * gamma_d ...
        - 0.5 * mp * Lp * Lr * sin(phi) * phi_d^2;
%second eq.
        - Bp * phi_d ...
        + 0.25 * mp * Lp^2 * cos(phi) * sin(phi) * gamma_d^2 ...
        - 0.5 * mp * Lp * g * sin(phi)
    ];

  
    dd = A \ b;
      % accelerations
    gamma_dd = dd(1);
    phi_dd = dd(2);

    % Derivata dello stato
    dxdt = [gamma_d;
            gamma_dd;
           phi_d;
            phi_dd];
end

end


%Funzione del voltaggio
function Vm = Vm_fun(t)
     Vm = 5 * sin(2*pi*0.2*t);  %segnale sinusoidale
end

% function Vm = Vm_fun(t)
%      step 
%     if t >= 2 && t < 5
%         Vm = 0.5; 
%     else
%         Vm = 0;  
%     end
% end

