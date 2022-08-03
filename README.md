# Differentiable barotropic gyre model

Our goal with this repo is to find a way to add Enzyme to existing Oceananigans code, namely the barotropic gyre model. 

The specific steps to get to an adjoint-able barotropic gyre model are

1) Create a "forward step" function. Right now the barotropic gyre model in Oceananigans takes in some initial inputs, and then just runs the model forward in its entirety to return the fields $u,$ $v,$ and $\eta$ at the end, if understood correctly. Ideally, we would instead take just one forward step:
              $$f(\mathbf{u}(t), \mathbf{v}(t), \mathbf{\eta}(t)) = (\mathbf{u}(t + \Delta t), \mathbf{v}(t + \Delta t), \mathbf{\eta}(t + \Delta t) )$$
(not trying to say $f$ returns a vector, just returns the fields at the next time step.) 

2) Once we have that forward function, we just want to apply Enzyme to see if it works. This shouldn't be difficult at this point (hopefully).

3) Then we can run a trial sensitivity analysis. Namely, per Patrick's suggestion we can choose the cost function $$J(\mathbf{u}(t_f), \mathbf{v}(t_f)) = \mathbf{u}^2(t_f) + \mathbf{v}^2(t_f) = \sum_{j, k} u_{jk}^2 + v_{jk}^2$$ (i.e. we sum $u^2 + v^2$ at every point on the grid and add them all together, not sure if this grid is 2D or 3D, if 3D we also need to sum over the third dimension.) Then we can ask what happens to this quantity if we change the initial $u$ field a little bit. This will just be running the "adjoint equations" step-by-step backwards.

