clc; clear all;
%% Training dataset
% Parametri
mp = 0.024;     % massa pendolo [kg]
mr=0.093;       %massa braccio [kg]
Lr = 0.086;     % lunghezza braccio [m]
Lp = 0.128;     % lunghezza pendolo [m]
Jr = 5.72e-5;    % inerzia braccio [kg m^2]
Jp = 1.31e-4;   % inerzia pendolo [kg m^2]
Br = 3e-4;    % attrito braccio [Nms]
Bp = 5e-4;   % attrito pendolo [Nms]
g  = 9.81;    % gravità [m/s^2]
Rm= 7.5; %ohm, resistenza interna
km=0.0431; %Motor back-emf constant

numInitialConditionPoints  = 400; 
% temporal points
t = linspace(0,15,15000);
Vm = arrayfun(@(ti) Vm_fun(ti), t);

% Condizioni iniziali:
t0IC = zeros(1,numInitialConditionPoints);
Vm0IC = zeros(1,numInitialConditionPoints);
phi0 = ones(1,numInitialConditionPoints)*(pi/6);
phi0_d=ones(1,numInitialConditionPoints)*0;
gamma0=ones(1,numInitialConditionPoints)*(0);
gamma0_d=ones(1,numInitialConditionPoints)*0;

%TRAINING DATASET
inputdata=[t ;Vm];
inputdataIC = [t0IC ; Vm0IC];
% [Angolo iniziale cart, velocità angolare iniziale cart, angolo iniziale pendolo, velocità angolare iniziale pendolo]
outputdataIC = [gamma0; gamma0_d; phi0; phi0_d];

%soluzione numerica
[t_num,x] = pendolo_soluzioneNumerica;
carrello=x(:,1);
pendolo=x(:,3);
t_num=t_num(:,1);

%% Neural Network
inputsize=2; %t, Vm
outputsize=2; %gamma, phi, 
hiddenunits=50;
numLayers = 8; 


% Input layer
layers = [featureInputLayer(inputsize, 'Name', 'input')];
% Hidden layers
for i = 2:numLayers-1
    layers = [layers 
              fullyConnectedLayer(hiddenunits, 'Name', ['fc' num2str(i)]) 
                % tanhLayer('Name', ['tanh' num2str(i)])];
                 % sigmoidLayer('Name', ['sigmoid' num2str(i)]);
                 % SineLayer(['Sine Activation Layer' num2str(i)])];
                  geluLayer('Name', ['gelu' num2str(i)])];
end


% Output layer
layers = [layers 
          fullyConnectedLayer(outputsize, 'Name', 'output')]; 

%Deep learning net
net = dlnetwork(layers);
 % deepNetworkDesigner(net)


%% Train the NN
%with ADAM optimizer
%ADAM hyperparameter
learnrate=1e-2;

meanp=[]; %mean term
vp=[]; %variance term
numinterations=3000;
bestLoss = inf;  
bestNet = net;   

%Training dataset in dlarray
inputdata=dlarray(inputdata, 'CB');
inputdataIC=dlarray(inputdataIC, 'CB');
outputdataIC=dlarray(outputdataIC, 'CB');
t=dlarray(t, 'CB');
Vm=dlarray(Vm, 'CB');
t0IC = dlarray(t0IC, 'CB');
gamma0 = dlarray(gamma0, 'CB');
gamma0_d=dlarray(gamma0_d, 'CB');
phi0=dlarray(phi0, 'CB');
phi0_d=dlarray(phi0_d, 'CB');

mp = dlarray(mp, 'CB');
Lp = dlarray(Lp, 'CB');
Jp = dlarray(Jp, 'CB');
Bp = dlarray(Bp, 'CB');
mr = dlarray(mr, 'CB');
Lr = dlarray(Lr, 'CB');
Jr = dlarray(Jr, 'CB');
Br = dlarray(Br, 'CB');
km=dlarray(km, 'CB');
Rm=dlarray(Rm, 'CB');
g=dlarray(g, 'CB');


%training progress plot
monitor= trainingProgressMonitor(Metrics=["Loss","LossPDE", "LossIC", "LossDATA", "GradNorm", "RMSE_phi", "RMSE_gamma"]);

%Acceleration del model loss
accFnc= dlaccelerate(@modelLoss); %LOSS FUNCTION

for iteration=1:numinterations
    %differenziazione automatica
    
    [loss, grad, lossPinn, lossIC, lossData, gamma, phi, gamma_d, phi_d, gamma_dd, phi_dd, tau]=dlfeval(accFnc, net, t, inputdata, inputdataIC, outputdataIC, mp, mr, Lp,Lr,Jp,Jr,Bp,Br,g, Rm, km); 

  %Salva come rete quella con loss inferiore
     if loss < bestLoss
        bestLoss = loss;
        bestNet = net;
     end

  
    %update the model
    [net,meanp,vp]=adamupdate(net,grad,meanp,vp, iteration, learnrate);

gradNorm = 0;
for i = 1:height(grad)
    AA = grad.Value{i};  % accedi al tensore
    if isnumeric(AA) && ~isempty(AA)
        gradNorm = gradNorm + sum(AA.^2, 'all');
    end
end

% Calcola la norma euclidea totale
gradNorm = sqrt(gradNorm);


 % Previsione training
    gamma_phi_tr = predict(net, inputdata);  
    gamma_pred_tr = gamma_phi_tr(1,:);
    phi_pred_tr = gamma_phi_tr(2,:);

    % Calcolo errori di training
    err_phi_training = phi_pred_tr - pendolo';
    err_gamma_training = gamma_pred_tr - carrello';
    rmse_phi_tr = sqrt(mean(err_phi_training.^2));
    rmse_gamma_tr = sqrt(mean(err_gamma_training.^2));
    
    fprintf('Iterazione %d - Gradiente Norm: %.6f - RMSE φ: %.4f - RMSE γ: %.4f\n', ...
        iteration, gradNorm, rmse_phi_tr, rmse_gamma_tr);

    % Salva metriche nel monitor
    recordMetrics(monitor, iteration, ...
        Loss=loss, ...
        LossPDE=lossPinn, ...
        LossIC=lossIC, ...
        LossDATA=lossData, ...
        GradNorm=gradNorm, ...
        RMSE_phi=rmse_phi_tr, ...
        RMSE_gamma=rmse_gamma_tr); 
end


net = bestNet;
loss=bestLoss;

disp(['MSE tot in training: ', num2str(loss)]);
disp(['MSE IC in training: ', num2str(lossIC)]);
disp(['MSE PINN in training: ', num2str(lossPinn)]);
 disp(['MSE DATA in training: ', num2str(lossData)]);


%% GRAFICI
ttest=[0:0.002:15];
Vmtest = arrayfun(@(ti) Vm_fun(ti), ttest);
inputtest=[ttest; Vmtest];
inputtest=dlarray(inputtest, 'CB');
gamma_phi=predict(net,inputtest);
gamma_pred=gamma_phi(1,:);
phi_pred=gamma_phi(2,:);

figure;
plot(ttest, phi_pred); 
hold on
plot(ttest, gamma_pred);
hold on
plot(t_num,carrello);
hold on
plot(t_num,pendolo);
hold on
plot(ttest, Vmtest);
xlabel('t'); ylabel('rad');
legend('phi-pendolo-pred', 'gamma-carrello-pred', 'carrello-numSolution', 'pendolo-numSolution', "Vm")

%errori di validazione
% Interpola la soluzione numerica sullo stesso tempo di ttest
pendolo_interp  = interp1(t_num, pendolo, ttest, 'linear', 'extrap');
carrello_interp = interp1(t_num, carrello, ttest, 'linear', 'extrap');

% Calcola gli errori
err_phi = phi_pred - pendolo_interp;
err_gamma = gamma_pred - carrello_interp;

% Calcola RMSE (Root Mean Square Error)
rmse_phi = sqrt(mean(err_phi.^2));
rmse_gamma = sqrt(mean(err_gamma.^2));

 %errori training
gamma_phi_tr=predict(net,inputdata);
gamma_pred_tr=gamma_phi_tr(1,:);
phi_pred_tr=gamma_phi_tr(2,:);
err_phi_training = phi_pred_tr - pendolo';
err_gamma_training = gamma_pred_tr - carrello';
% Calcola RMSE (Root Mean Square Error), opzionale
rmse_phi_tr = sqrt(mean(err_phi_training.^2));
rmse_gamma_tr = sqrt(mean(err_gamma_training.^2));


% Mostra i risultati
fprintf("RMSE TRAINING φ (pendolo): %.4f \n", rmse_phi_tr);
fprintf("RMSE TRAINING γ (carrello): %.4f \n", rmse_gamma_tr);
fprintf("RMSE validation φ (pendolo): %.4f \n", rmse_phi);
fprintf("RMSE validation γ (carrello): %.4f \n", rmse_gamma);

%% LOSS FUNCTION
function [loss, grad, lossPinn, lossIC, lossData, gamma, phi, gamma_d, phi_d, gamma_dd, phi_dd, tau]=modelLoss(net, t, inputdata, inputdataIC,outputdataIC,mp, mr, Lp,Lr,Jp,Jr,Bp,Br,g, Rm, km); 

[lossPinn,  gamma, phi, gamma_d, phi_d, gamma_dd, phi_dd, tau]= pinnsLoss(net, t, inputdata, mp, mr, Lp,Lr,Jp,Jr,Bp,Br,g, Rm, km);


%Loss initial condition
lossIC = ICloss(net, inputdataIC, outputdataIC, t0IC);

%Loss data
[tloss,x]=pendolo_soluzioneNumerica; %tloss= [t, Vm]
tloss=tloss';
tloss=dlarray(tloss, 'CB');
gamma_phi = forward(net, tloss);
gamma_data=gamma_phi(1,:);
phi_data=gamma_phi(2,:);
carrello=x(:,1)';
pendolo=x(:,3)';
lossData = mean((gamma_data - carrello).^2) + mean((phi_data - pendolo).^2);

    

%Total loss and gradients
loss=lossPinn+lossIC+lossData;
grad=dlgradient(loss, net.Learnables);
end


%% PINN loss function
function [lossPinn, gamma, phi, gamma_d, phi_d, gamma_dd, phi_dd, tau]= pinnsLoss(net, t, inputdata, mp, mr, Lp,Lr,Jp,Jr,Bp,Br,g, Rm, km);
gamma_phi = forward(net, inputdata);
Vm=inputdata(2,:);
gamma=gamma_phi(1,:);
phi=gamma_phi(2,:);

% First order derivatives
gamma_d = dlgradient(sum(gamma, 'all'), inputdata, 'EnableHigherDerivatives', true); %velocità carrello
gamma_d=gamma_d(1,:);
phi_d = dlgradient(sum(phi, 'all'), inputdata, 'EnableHigherDerivatives', true); %velocità pendolo
phi_d=phi_d(1,:);

%Second order derivatives
   gamma_dd = dlgradient(sum(gamma_d, 'all'), inputdata, 'EnableHigherDerivatives', true); %accelerazione carrello
   gamma_dd=gamma_dd(1,:);
    phi_dd = dlgradient(sum(phi_d, 'all'), inputdata, 'EnableHigherDerivatives', true); %accelerazione pendolo
phi_dd=phi_dd(1,:);

%Coppia motore
tau=km.*((Vm-km.*gamma_d)./Rm);

%Residuo cart
R_gamma=(mp .* (Lr.^2) + 0.25 .* mp .* (Lp.^2) - 0.25 .* mp .* (Lp.^2) .* cos(phi).^2 + Jr) .* gamma_dd ...
  - 0.5 .* mp .* Lp .* Lr .* cos(phi) .* phi_dd ...
  + 0.5 .* mp .* (Lp.^2) .* sin(phi) .* cos(phi) .* phi_d .* gamma_d ...
  +0.5 .* mp .* Lp .* Lr .* sin(phi) .* phi_d.^2 ...
  - tau + Br .* gamma_d;

%Residuo pendolo
R_phi = 0.5 .* mp .* Lp .* Lr .* cos(phi) .* gamma_dd ...
  + (Jp + 0.25 .* mp .* (Lp.^2)) .* phi_dd ...
  - 0.25 .* mp .* (Lp.^2) .* cos(phi) .* sin(phi) .* gamma_d.^2 ...
  + 0.5 .* mp .* Lp .* g .* sin(phi) ...
  + Bp .* phi_d;


% MSE
lossPinn_phi = mean(R_phi.^2, 'all');  
lossPinn_gamma= mean(R_gamma.^2, 'all'); 
lossPinn = lossPinn_phi+lossPinn_gamma;

end


 

%%  INITIAL CONDITION LOSS FUNCTION
function lossIC = ICloss(net, inputdataIC, outputdataIC)
%outputdataIC = [gamma0; gamma0_d; phi0; phi0_d];
  gamma0=outputdataIC(1,:);
  gamma0_d=outputdataIC(2,:);
  phi0=outputdataIC(3,:);
  phi0_d=outputdataIC(4,:);

    % initial learned from network
gamma_phi_pred = forward(net, inputdataIC); 
gamma0_pred = gamma_phi_pred(1,:); %carrello
phi0_pred = gamma_phi_pred(2,:);%pendolo


gamma0d_pred = dlgradient(sum(gamma0_pred), inputdataIC, 'EnableHigherDerivatives', true); %velocità carrello
gamma0d_pred=gamma0d_pred(1,:);
phi0d_pred = dlgradient(sum(phi0_pred), inputdataIC, 'EnableHigherDerivatives', true); %velocità pendolo
phi0d_pred=phi0d_pred(1,:);
    
    %MSE
    lossIC_gamma = mean(( gamma0_pred - gamma0).^2, 'all');  
    lossIC_phi = mean(( phi0_pred - phi0).^2, 'all');
    lossIC_gammad = mean(( gamma0d_pred - gamma0_d).^2, 'all');
    lossIC_phid = mean(( phi0d_pred - phi0_d).^2, 'all');
    lossIC=lossIC_gamma+lossIC_phi+lossIC_gammad+lossIC_phid;
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
