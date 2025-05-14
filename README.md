# Inverted-pendulum---PINN
# Inverted Pendulum on Circular Cart (Quanser) 

Physics-Informed Neural Network (PINN) model for simulating the inverted pendulum on a circular cart.

The equations and parameters used in this project are based on the file ---.

## Project Structure

The code is organized into the following sections:

- **Training dataset creation**
- **Neural network (NN) architecture definition**
- **NN training using ADAM optimizer**
- **Training plots**
- **Validation plots**
- **Error analysis**

## Implemented Functions

- `model_loss`: main function for computing the total loss  
- `pinn_loss`: PDE-based loss function  
- `lossIC`: loss for initial condition enforcement  
- `pendolo_soluzioneNumerica`: numerical solution of the system's differential equations.  
  Outputs time, input voltage, cart angle, and pendulum angle:  
  ```matlab
  [t, x] = [(t, Vm_values), (gamma, phi)]
- `Vm_fun`: defines the input voltage signal
- `pendulum_ODE`: defines the system's ODEs, computes angular accelerations of the cart and pendulum
