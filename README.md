# Solving RWA (Routing and Wavelength Allocation) using QUBO

### DISCLAMER: The full-source branch contains extra code used in early development stages. It is no longer maintained and likely contains sloppy/unused code. It is not recommended to use it as-is. Pull requests with fixes are welcome.

Source code for my dissertation on using quantum annealing for solving wavelength allocation and routing and wavelength allocation problems in optical networks. Includes a QUBO formulation of the RWA problem using precompiled routes.

The main code is in the .ipynb file, and the topologies are in the topologies directory. Look at the ConnectedNetwork class for more info on how the topology files are loaded and handled.
