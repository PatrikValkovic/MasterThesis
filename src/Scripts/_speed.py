###############################
#
# Created by Patrik Valkovic
# 4/3/2021
#
###############################
import torch as t
import numpy as np
import time

REPEATS = 1000
PARENTS = 10000
TO_PICK = 1
CHILDREN = 5000

# init
t.set_num_threads(4)
d = t.device('cuda:0')
t.rand(1,device=d)

print("=== CPU ===")

# randint
bufferp, bufferf = [], []
s = time.time()
for _ in range(REPEATS):
    startp = time.process_time()
    startf = time.perf_counter()
    x = t.randint(PARENTS, (CHILDREN, TO_PICK), dtype=t.long)
    float(x.max())
    endp = time.process_time()
    endf = time.perf_counter()
    bufferp.append(endp - startp)
    bufferf.append(endf - startf)
e = time.time()
##############################
print(f"randint               p:{np.mean(bufferp):.6f} f:{np.mean(bufferf):.6f} in reality {e-s:.6f}")

# rand permutation
bufferp, bufferf = [], []
s = time.time()
for _ in range(REPEATS):
    startp = time.process_time()
    startf = time.perf_counter()
    x = t.randperm(PARENTS, dtype=t.long)[:CHILDREN]
    float(x.max())
    endp = time.process_time()
    endf = time.perf_counter()
    bufferp.append(endp - startp)
    bufferf.append(endf - startf)
e = time.time()
##############################
print(f"randperm              p:{np.mean(bufferp):.6f} f:{np.mean(bufferf):.6f} in reality {e-s:.6f}")

# uniform
bufferp, bufferf = [], []
s = time.time()
for _ in range(REPEATS):
    startp = time.process_time()
    startf = time.perf_counter()
    x = t.rand((CHILDREN, TO_PICK))
    x.multiply_(PARENTS)
    x = x.to(t.long)
    float(x.max())
    endp = time.process_time()
    endf = time.perf_counter()
    bufferp.append(endp - startp)
    bufferf.append(endf - startf)
e = time.time()
##############################
print(f"uniform               p:{np.mean(bufferp):.6f} f:{np.mean(bufferf):.6f} in reality {e-s:.6f}")

# multinomial
bufferp, bufferf = [], []
probs = t.tensor(1 / PARENTS).as_strided_((CHILDREN,PARENTS),(0,0))
s = time.time()
for _ in range(REPEATS):
    startp = time.process_time()
    startf = time.perf_counter()
    x = t.multinomial(probs, TO_PICK)
    float(x.max())
    endp = time.process_time()
    endf = time.perf_counter()
    bufferp.append(endp - startp)
    bufferf.append(endf - startf)
e = time.time()
##############################
print(f"multinomial repeat    p:{np.mean(bufferp):.6f} f:{np.mean(bufferf):.6f} in reality {e-s:.6f}")

bufferp, bufferf = [], []
probs = t.tensor(1 / PARENTS).as_strided_((CHILDREN,PARENTS),(0,0))
s = time.time()
for _ in range(REPEATS):
    startp = time.process_time()
    startf = time.perf_counter()
    x = t.multinomial(probs, TO_PICK, replacement=False)
    float(x.max())
    endp = time.process_time()
    endf = time.perf_counter()
    bufferp.append(endp - startp)
    bufferf.append(endf - startf)
e = time.time()
##############################
print(f"multinomial no repeat p:{np.mean(bufferp):.6f} f:{np.mean(bufferf):.6f} in reality {e-s:.6f}")




print("=== GPU ===")

# randint
bufferp, bufferf = [], []
s = time.time()
for _ in range(REPEATS):
    startp = time.process_time()
    startf = time.perf_counter()
    x = t.randint(PARENTS, (CHILDREN, TO_PICK), dtype=t.long, device=d)
    float(x.max())
    endp = time.process_time()
    endf = time.perf_counter()
    bufferp.append(endp - startp)
    bufferf.append(endf - startf)
e = time.time()
##############################
print(f"randint               p:{np.mean(bufferp):.6f} f:{np.mean(bufferf):.6f} in reality {e-s:.6f}")

# rand permutation
bufferp, bufferf = [], []
s = time.time()
for _ in range(REPEATS):
    startp = time.process_time()
    startf = time.perf_counter()
    x = t.randperm(PARENTS, dtype=t.long, device=d)[:CHILDREN]
    float(x.max())
    endp = time.process_time()
    endf = time.perf_counter()
    bufferp.append(endp - startp)
    bufferf.append(endf - startf)
e = time.time()
##############################
print(f"randperm              p:{np.mean(bufferp):.6f} f:{np.mean(bufferf):.6f} in reality {e-s:.6f}")

# uniform
bufferp, bufferf = [], []
s = time.time()
for _ in range(REPEATS):
    startp = time.process_time()
    startf = time.perf_counter()
    x = t.rand((CHILDREN, TO_PICK), device=d)
    x.multiply_(PARENTS)
    x = x.to(t.long)
    float(x.max())
    endp = time.process_time()
    endf = time.perf_counter()
    bufferp.append(endp - startp)
    bufferf.append(endf - startf)
e = time.time()
##############################
print(f"uniform               p:{np.mean(bufferp):.6f} f:{np.mean(bufferf):.6f} in reality {e-s:.6f}")

# multinomial
bufferp, bufferf = [], []
probs = t.tensor(1 / PARENTS, device=d).as_strided_((CHILDREN,PARENTS),(0,0))
s = time.time()
for _ in range(REPEATS):
    startp = time.process_time()
    startf = time.perf_counter()
    x = t.multinomial(probs, TO_PICK)
    float(x.max())
    endp = time.process_time()
    endf = time.perf_counter()
    bufferp.append(endp - startp)
    bufferf.append(endf - startf)
e = time.time()
##############################
print(f"multinomial repeat    p:{np.mean(bufferp):.6f} f:{np.mean(bufferf):.6f} in reality {e-s:.6f}")

bufferp, bufferf = [], []
probs = t.tensor(1 / PARENTS, device=d).as_strided_((CHILDREN,PARENTS),(0,0))
s = time.time()
for _ in range(REPEATS):
    startp = time.process_time()
    startf = time.perf_counter()
    x = t.multinomial(probs, TO_PICK, replacement=False)
    float(x.max())
    endp = time.process_time()
    endf = time.perf_counter()
    bufferp.append(endp - startp)
    bufferf.append(endf - startf)
e = time.time()
##############################
print(f"multinomial no repeat p:{np.mean(bufferp):.6f} f:{np.mean(bufferf):.6f} in reality {e-s:.6f}")
