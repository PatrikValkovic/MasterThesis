import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import copy
import scipy.stats as stats

# move to the current directory
os.chdir(os.path.dirname(sys.argv[0]))
FIGSIZE=6.4, 4.8
exit()

# differential evolution example
gen = np.random.default_rng(42)
fn = lambda x,y: x**2+3*y**2+2*x*y
x = np.linspace(-3,3,1000)
y = np.linspace(-3,3,1000)
X, Y = np.meshgrid(x,y)
Z = fn(X,Y)
plt.figure(figsize=FIGSIZE)
contours = plt.contour(X,Y,Z, levels=[0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 6.0, 12.0], colors='black', alpha=0.2)
pop = gen.multivariate_normal([0,0],[[0.7, -0.5],[-0.5, 0.7]], size=16)
plt.scatter(pop[:,0],pop[:,1], c='tab:blue', marker='x', alpha=0.5, label='population', s=32)
p1, p2, p3, F = 5, 11, 15, 0.7
plt.arrow(pop[p3,0], pop[p3,1], pop[p2,0] - pop[p3,0], pop[p2,1] - pop[p3,1], color='black', width=0.01, length_includes_head=True, head_width=0.1)
plt.text(-1.4,1.35,'$p_2 - p_3$', fontsize=14)
vect = F*(pop[p2] - pop[p3])
plt.arrow(pop[p1,0], pop[p1,1], vect[0], vect[1] , color='black', width=0.01, length_includes_head=True, head_width=0.1)
plt.text(-0.1,0.8,'$p_1 + F(p_2 - p_3)$', fontsize=14)
new_inv = pop[p1] + vect
plt.scatter(pop[p1,0], pop[p1,1], c='tab:green', marker='x', label='parents', zorder=5, s=80, linewidth=3)
plt.scatter(pop[p2,0], pop[p2,1], c='tab:green', marker='x', zorder=5, s=80, linewidth=3)
plt.scatter(pop[p3,0], pop[p3,1], c='tab:green', marker='x', zorder=5, s=80, linewidth=3)
plt.scatter(new_inv[0],new_inv[1], c='tab:orange', marker='x', label='new individual', zorder=5, s=80, linewidth=3)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-2,2)
plt.ylim(-1,2)
plt.legend()
plt.savefig('render_differential.pdf')
plt.close()

# decays
STEPS = 400
x = np.arange(0.0, STEPS + 1)
plt.figure(figsize=FIGSIZE)
plt.plot(x, 5 - 5 * (x / STEPS), c='tab:blue', label='Linear')
plt.plot(x, 5 * 0.9**x, c='tab:orange', label='Exponential $r=0.9$')
plt.plot(x, 5 * 0.99**x, c='tab:green', label='Exponential $r=0.99$')
plt.plot(x, 5 * (1 - x/STEPS)**2, c='tab:red', label='Polynomial $p=2$')
plt.plot(x, 5 * (1 - x/STEPS)**7, c='tab:purple', label='Polynomial $p=7$')
plt.legend()
plt.savefig('render_decayrate.pdf')


# normal, cauchy distributions
x = np.linspace(-3,3,1000)
plt.figure(figsize=FIGSIZE)
plt.plot(x, stats.norm.pdf(x, 0, 1), c='tab:blue', label="$N(0,1)$")
plt.plot(x, stats.cauchy.pdf(x, 0, 1), c='tab:orange', label="Cauchy $t=1$")
plt.plot(x, stats.cauchy.pdf(x, 0, 1.5), c='tab:green', label="Cauchy $t=1.5$")
plt.legend()
plt.savefig('render_distributions_normalcauchy.pdf')
plt.close()

x = np.linspace(-1,1,1000)
plt.figure(figsize=FIGSIZE)
n = 2
plt.plot(x, 0.5 * (n+1) * (1-np.abs(x))**n, c='tab:blue', label='Polynomial $n=2$')
n = 3
plt.plot(x, 0.5 * (n+1) * (1-np.abs(x))**n, c='tab:orange', label='Polynomial $n=3$')
n = 5
plt.plot(x, 0.5 * (n+1) * (1-np.abs(x))**n, c='tab:green', label='Polynomial $n=5$')
plt.legend()
plt.savefig('render_distributions_polynomial.pdf')
plt.close()

# render elipsoids
fn = lambda x,y: x**2+3*y**2
x = np.linspace(-3,3,1000)
y = np.linspace(-3,3,1000)
X, Y = np.meshgrid(x,y)
Z = fn(X,Y)
stepsize = 0.02
individuals = [np.array([2.0,2.0])]
optimizing_index = 0
while not np.all(np.abs(individuals[-1]) < 1e-6):
    cf = fn(individuals[-1][0],individuals[-1][1])
    n = copy.deepcopy(individuals[-1])
    n[optimizing_index] += stepsize
    f = fn(n[0], n[1])
    if f < cf:
        individuals.append(n)
        continue
    n = copy.deepcopy(individuals[-1])
    n[optimizing_index] -= stepsize
    f = fn(n[0], n[1])
    if f < cf:
        individuals.append(n)
        continue
    optimizing_index = 1 - optimizing_index
individuals = np.stack(individuals)
print(f"STEPS separable: {len(individuals)}")
plt.figure(figsize=FIGSIZE)
contours = plt.contour(X,Y,Z, levels=[0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 6.0, 12.0], colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.plot(individuals[:,0], individuals[:,1], c='tab:orange', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_axisbelow(True)
plt.gca().yaxis.grid(color='gray', linestyle=':')
plt.gca().xaxis.grid(color='gray', linestyle=':')
plt.savefig("render_separable.pdf")
plt.close()

fn = lambda x,y: x**2+3*y**2+2*x*y
x = np.linspace(-3,3,1000)
y = np.linspace(-3,3,1000)
X, Y = np.meshgrid(x,y)
Z = fn(X,Y)
stepsize = 0.02
individuals = [np.array([2.0,2.0])]
optimizing_index = 0
while not np.all(np.abs(individuals[-1]) < 1e-6):
    cf = fn(individuals[-1][0],individuals[-1][1])
    n = np.copy(individuals[-1])
    n[optimizing_index] += stepsize
    f = fn(n[0], n[1])
    if f < cf:
        individuals.append(n)
        continue
    n = np.copy(individuals[-1])
    n[optimizing_index] -= stepsize
    f = fn(n[0], n[1])
    if f < cf:
        individuals.append(n)
        continue
    optimizing_index = 1 - optimizing_index
individuals = np.stack(individuals)
print(f"STEPS non separable: {len(individuals)}")
plt.figure(figsize=FIGSIZE)
contours = plt.contour(X,Y,Z, levels=[0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 6.0, 12.0], colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.plot(individuals[:,0], individuals[:,1], c='tab:orange', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_axisbelow(True)
plt.gca().yaxis.grid(color='gray', linestyle=':')
plt.gca().xaxis.grid(color='gray', linestyle=':')
plt.savefig("render_nonseparable.pdf")
plt.close()
print()