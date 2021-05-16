# Master Thesis

Title: GPU Paralellization of Evlutionary Algorithms

Author: Patrik Valkovič

Department: Deparment of Theoretical Computer Science and Mathematical Logic

Supervisor: Mgr. Martin Pilát, Ph.D.

Abstract: Graphical Processing Units stand for the success of Artificial Neural Networks over the past decade and their broader application in the industry.
Another promising field of Artificial Intelligence is Evolutionary Algorithms.
Their parallelization ability is well known and has been successfully applied in practice.
However, these attempts focused on multi–core and multi–machine parallelization rather than on the GPU.

This work explores the possibilities of Evolutionary Algorithms parallelization on GPU.
I propose implementation in PyTorch library, allowing to execute EA on both CPU and GPU.
The proposed implementation provides the most common evolutionary operators for Genetic Algorithms, Real–Coded Evolutionary Algorithms, and Particle Swarm Optimization Algorithms.
Finally, I show the performance is an order of magnitude faster on GPU for medium and big–sized problems and populations.


## Content

```text
.
+--src - source codes used for this work
|  +--BBOBtorch - implementation of BBOB functions in PyTorch
|  +--FFEAT - implementation of FFEAT library
|  +--Scripts - scripts used for evaluation and measurement
|  `--Examples - some examples of how to use the FFEAT library
+--thesis - thesis in the LaTeX format
|  +--img - figures used in this thesis
|  `--measurements - measurements in higher resolution
+--thesis.pdf - digital version of this thesis
`--README.md - brief description of the content
```

--------------------

License: MIT
