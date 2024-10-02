# Datasets

This folder in mainly concerned with two main tasks involving data:

- preprocessing existing binary datasets into torch Datasets so that they are appropriate
for training. The functions concerning this preprocessing can be found in the file
`binary_datasets.py`;
- Create some analytical datasets for which we know explicit gradients, so that
we can perform a number of controlled tests. These can be found in the file
`toy_datasets.py`.