Scripts used to run and measure experiments

- `cpp` directory contains C++ implementation of the Genetic Algorithm.
- `problems` directory contains SAT fitness function.
- `utils` directory contains helper functions to measure the execution time and generate SAT problems.
- files prefixed with underscore are used to render statistics about the experiments or modify them.
- files prefixed with `fitness_` are scripts used to measure algorithm fitness.
- files prefixed with `time_` are scripts used to measure running time of the algorithm.
- files prefixed with `timefitness_` are scripts used to measure algorithm fitness and its running time.
- Python files prefixed with `sweep_` are scripts used to run hyperparameter search.
- YAML files prefixed with `sweep_` are configuration files to run hyperparameter search.
- files prefixed with `generate_` are scripts that generates bash scripts. These bash scripts were use afterwards to start experiments on MetaCentrum.
