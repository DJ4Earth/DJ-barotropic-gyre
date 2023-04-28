# Differentiable ocean model

We aim to create a differentiable barotropic gyre model (i.e. just a simple ocean model) in the Julia ecosystem. At this point there are two threads:

  1. Using the Julia packages Oceananigans and Enzyme. Oceananigans is an existing, efficient Julia package that contains all the code we need for a barotropic gyre model, at which point we would just differentiate this code. The files barotropic_gyre_exp.jl, barotropic_gyre_original.jl, and barotropic_gyre_singlestep.jl are explicit barotropic gyre solvers written using Oceananigans functions (thanks to Greg Wagner for creating these!). We have yet to add Enzyme successfully, but will continue to work on this. 
  
  2. The second direction is a different barotropic gyre model using a fully explicit solver (in this case RK4). This code lives in the folder explicit_solver, and is based on Python code written by Milan Kloewer (found here: https://github.com/milankl/swm). 
  
The explicit solver code is currently working with both Enzyme and Checkpointing. The file main_energy_1_chkp.jl runs a sample adjoint problem and computes the sensitivity of the final energy with respect to the initial conditions. 
