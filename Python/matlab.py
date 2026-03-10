import scipy.io
mat = scipy.io.loadmat('BlueNoiseExperimentWithSpots_150StepSize_512x512x1000.mat')
print(mat.keys())  # List all variable names in the .mat file
print(mat['s'], mat['sss'], mat['ii'], mat['iiStep'], mat['iiStepI'])