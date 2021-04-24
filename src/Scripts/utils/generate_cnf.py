###############################
#
# Created by Patrik Valkovic
# 24.04.21
#
###############################
from typing import List
import numpy as np
import time
import argparse


def generate_cnf(literals: int, clauses: List[int], f: str = 'tmp.cnf'):
    sample = np.random.rand(literals) < 0.5
    with open(f, 'w') as f:
        print(f'c generated for purpose of FFEAT evaluation', file=f)
        print(f'c {time.asctime()}', file=f)
        print(f'c SOLUTION: {str.join(" ", map(lambda x: "1" if x else "0", sample))}', file=f)
        print(f'p cnf {literals} {len(clauses)}', file=f)
        for clause_c in clauses:
            selected = np.random.choice(literals, clause_c, replace=False)
            values = sample[selected]
            selected += 1
            selected[np.logical_not(values)] *= -1
            print(str.join(" ", map(str, selected)), file=f)


def generate_cnf_norm(literals: int, mclauses, sclauses, mperclause, sperclause, f: str = 'tmp.cnf'):
    clauses = max(int(np.random.randn() * sclauses + mclauses), 1)
    clauses = [
        max(int(np.random.randn() * sperclause + mperclause), 1)
        for _ in range(clauses)
    ]
    return generate_cnf(literals, clauses,f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='tmp.cnf', help='Output file')
    parser.add_argument('--literals', type=int, default=100, help='Number of literals')
    parser.add_argument('--mean_clauses', type=float, default=260.0, help='Expected number of clauses')
    parser.add_argument('--std_clauses', type=float, default=20.0, help='Deviation of number of clauses')
    parser.add_argument('--mean_literals_per_clause', type=float, default=7.0, help='Expected number of literals in clause')
    parser.add_argument('--std_literals_per_clause', type=float, default=4.0, help='Deviation of number of literals per clause')
    args, _ = parser.parse_known_args()
    generate_cnf_norm(
        args.literals,
        args.mean_clauses,
        args.std_clauses,
        args.mean_literals_per_clause,
        args.std_literals_per_clause,
        args.output
    )
