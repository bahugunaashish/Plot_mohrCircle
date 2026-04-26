# Plot_mohrCircle
Interactive Mohr-Coulomb failure envelope analyser for consolidated-undrained triaxial tests with live plotting GUI.


## Features
- Add unlimited triaxial test data (chamber pressure, deviator stress, pore pressure)
- Automatically computes effective principal stresses
- Best-fit common tangent to all Mohr circles using least-squares geometry
- Calculates shear strength parameters c' and φ'
- Estimates shear strength at any depth in the soil layer
- Live plot updates instantly as you type
- Built-in zoom, pan, and save tools via matplotlib toolbar
- Clean light-themed GUI built with Tkinter

## Requirements
- Python 3.7+
- numpy
- matplotlib

## Screenshot
![App Screenshot](https://raw.githubusercontent.com/bahugunaashish/Plot_mohrCircle/main/Screenshot.png)
