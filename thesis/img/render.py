import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import copy

# move to the current directory
os.chdir(os.path.dirname(sys.argv[0]))

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
plt.figure(figsize=(8,6))
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
plt.figure(figsize=(8,6))
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
