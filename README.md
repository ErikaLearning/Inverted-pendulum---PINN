# Inverted-pendulum---PINN
# Inverted Pendulum on Circular Cart (Quanser) 

Physics-Informed Neural Network (PINN) model for simulating the inverted pendulum on a circular cart.

The equations and parameters used in this project are based on the file System_identification_of_a_2DOF_pe .

## Code Structure

The code is organized into the following sections:

- **Training dataset creation**
- **Neural network (NN) architecture definition**
- **NN training using ADAM optimizer**
- **Validation plots**
- **Error analysis**

## Implemented Functions

The file `PINN_pendolo` includes the following main functions:

- `model_loss`: main function for computing the total loss  
- `pinn_loss`: PDE-based loss function  
- `lossIC`: loss for initial condition enforcement

The function `pendolo_soluzioneNumerica` calculates the numerical solution of the system's differential equations.  
It outputs time, input voltage, cart angle, and pendulum angle:  
```matlab
[t, x] = [(t, Vm_values), (gamma, phi)]
```

Within `pendolo_soluzioneNumerica`, the following helper functions are used:

- `Vm_fun`: defines the input voltage signal applied to the system  
- `pendulum_ODE`: defines the system's ordinary differential equations (ODEs).  
  It returns the angular velocities (first derivatives) and angular accelerations (second derivatives) of the cart and the pendulum:

  ```matlab
  dxdt = [gamma_d; gamma_dd; phi_d; phi_dd];
