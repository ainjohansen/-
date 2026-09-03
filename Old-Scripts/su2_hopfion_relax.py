import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse

# =========================
# CLI
# =========================

parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=200)
parser.add_argument("--dtau", type=float, default=5e-6)
parser.add_argument("--grid", type=int, default=50)
parser.add_argument("--workers", type=int, default=cpu_count())
args = parser.parse_args()

# =========================
# GRID
# =========================

Nr = args.grid
Nt = args.grid // 2

r = np.linspace(1e-3, 6.0, Nr)
t = np.linspace(1e-3, np.pi - 1e-3, Nt)

dr = r[1] - r[0]
dt = t[1] - t[0]

R, T = np.meshgrid(r, t, indexing='ij')

# =========================
# PARAMETERS
# =========================

m = 1
mu = 0.5
lam = 0.5
dtau = args.dtau
eps = 1e-4

# =========================
# INITIAL
# =========================

f = np.pi * np.exp(-R)
g = T.copy()

# =========================
# DERIVATIVES
# =========================

def grad(F):
    Fr = np.gradient(F, dr, axis=0)
    Ft = np.gradient(F, dt, axis=1)
    return Fr, Ft

# =========================
# ENERGY
# =========================

def energy(f, g):
    sinf = np.sin(f)
    sing = np.sin(g)

    fr, ft = grad(f)
    gr, gt = grad(g)

    E2 = fr**2 + ft**2
    E_sigma = sinf**2 * (gr**2 + gt**2)
    E_twist = sinf**2 * sing**2 * (m**2 / (R**2 * (np.sin(T)**2 + 1e-3)))

    sk = (fr * gt - ft * gr)**2
    E4 = lam * sinf**2 * sk

    V = mu**2 * (1 - np.cos(f))

    return np.sum((E2 + E_sigma + E_twist + E4 + V) * R**2 * np.sin(T))

# =========================
# PARALLEL GRADIENT
# =========================

def grad_point(args):
    i, j, f, g = args

    # f variation
    f_loc = f.copy()
    f_loc[i, j] += eps
    E_plus = energy(f_loc, g)

    f_loc[i, j] -= 2 * eps
    E_minus = energy(f_loc, g)

    dE_df = (E_plus - E_minus) / (2 * eps)

    # g variation
    g_loc = g.copy()
    g_loc[i, j] += eps
    E_plus = energy(f, g_loc)

    g_loc[i, j] -= 2 * eps
    E_minus = energy(f, g_loc)

    dE_dg = (E_plus - E_minus) / (2 * eps)

    return i, j, dE_df, dE_dg

# =========================
# RELAXATION
# =========================

print("=== START RELAXATION ===")
print(f"grid={Nr}x{Nt}, iters={args.iters}, workers={args.workers}")

pool = Pool(args.workers)

outer_bar = tqdm(range(args.iters), desc="Total progress")

for it in outer_bar:

    tasks = [(i, j, f, g) for i in range(Nr) for j in range(Nt)]

    results = list(tqdm(
        pool.imap(grad_point, tasks),
        total=len(tasks),
        leave=False,
        desc=f"iter {it}"
    ))

    dE_df = np.zeros_like(f)
    dE_dg = np.zeros_like(g)

    for i, j, dfv, dgv in results:
        dE_df[i, j] = dfv
        dE_dg[i, j] = dgv

    # update
    f -= dtau * dE_df
    g -= dtau * dE_dg

    # BC
    f[0, :] = np.pi
    f[-1, :] = 0

    g[0, :] = 0
    g[-1, :] = np.pi
    g[:, 0] = 0
    g[:, -1] = np.pi

    # clamp
    f = np.clip(f, 0, np.pi)
    g = np.clip(g, 0, np.pi)

    E = energy(f, g)

    outer_bar.set_postfix({
        "E": f"{E:.4f}",
        "f_max": f"{np.max(f):.2f}",
        "f_min": f"{np.min(f):.2f}"
    })

pool.close()
pool.join()

print("=== DONE ===")
