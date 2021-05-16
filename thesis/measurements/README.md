# Results of experiments

This directory contains two types of graphs:
- running time of the algorithm with respect to population size, prefixed with `time_`,
- fitness of the algorithm with respect to generations, prefixed with `fitness_`.

Overall, the directory includes following graphs:
- `fitness_es_crossover_f{fn}_dim{d}_{crossover}.pdf` - fitness of crossover operators where `{fn}` is the BBOB function, `d` is the dimension of the function, and `{crossover}` is the crossover type.
- `fitness_es_mutation_f{fn}_dim{d}_{mutation}.pdf` - fitness of mutation operators where `{fn}` is the BBOB function, `d` is the dimension of the function, and `{mutation}` is the mutation type.
- `fitness_es_schema_f{fn}_dim{d}_{schema}.pdf` - fitness of mutation schemes where `{fn}` is the BBOB function, `d` is the dimension of the function, and `{schema}` is the schema type.
- `fitness_ga_3SAT_d{d}.pdf` - fitness of Genetic Algorithm on 3SAT problem with $d$ literals.
- `fitness_ga_elitism_3SAT_d{d}.pdf` - fitness of Genetic Algorithm with elitism on 3SAT problem with $d$ literals.
- `fitness_pso_f{fn}_{neig}.pdf` - fitness of PSO2006 algorithm where `{fn}` is the BBOB function and `{neig}` is the neighborhood type.
- `fitness_pso{2006/2011}_f{fn}.pdf` - fitness of PSO2006 and PSO2011 algorithm where `{fn}` is the BBOB function.
- `time_es_crossover_fn{f}_{dim}d.pdf` - running time of crossover operators where `{f}` is the BBOB function and `{dim}` is its dimension.
- `time_es_mutation_fn{f}_{dim}d.pdf` - running time of mutation operators where `{f}` is the BBOB function and `{dim}` is its dimension.
- `time_es_schema_fn{f}_{dim}d.pdf` - running time of crossover schemes where `{f}` is the BBOB function and `{dim}` is its dimension.
- `time_ga.pdf` - running time of Genetic Algorithm.
- `time_ga_c.pdf` - running time of Genetic Algorithm implemented in C and PyTorch running on CPU and GPU.
- `time_ga_elitism.pdf` - running time of Genetic Algorithm with elitism.
- `time_ga_clausecount.pdf` and `time_ga_varcount.pdf` - running time of Genetic Algorithm with respect to problem size.
- `time_ga_scale_{lit}l.pdf` - running time of fitness scale operators. It uses Genetic Algorithm and the problem has `{lit}` literals.
- `time_ga_selections.pdf` - running time of selection operators.
- `time_pso{2006/2011}_fn{f}_alldim.pdf` - running time of PSO2006 and PSO2011 algorithm where `{f}` is the BBOB function.
- `time_pso2006_fn{f}_neigh.pdf` - running time of PSO2006 algorithm with various neighborhoods where `{f}` is the BBOB function.