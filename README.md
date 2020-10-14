# Econ-SIR

Econ-SIR model implementation using JAX. Works best with Python 3.8. Simplified version of model used in:

[Economics and Epidemics: Evidence from an Estimated Spatial Econ-SIR Model](http://doughanley.com/files/papers/COVID.pdf) (with Mark Bognanni, Daniel Kolliner, and Kurt Mitman)

For a live dashboard of the more complex model in the above paper see: [econsir.io](econsir.io)

## Usage

Check out `econsir.ipynb` for a quick walkthrough of simulation, estimation, and optimal policy.

To run the built-in dashboard, execute
```
python3 dash.py
```

Here's an example of a simulated path

![simulated path](simul.svg)
