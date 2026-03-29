import numpy as np
from scipy.integrate import dblquad

# Параметризация трилистника (стандартная)
def r(t):
    t = np.asarray(t)
    x = np.sin(t) + 2*np.sin(2*t)
    y = np.cos(t) - 2*np.cos(2*t)
    z = -np.sin(3*t)
    return np.vstack([x, y, z]).T  # форма (N,3)

def dr(t):
    t = np.asarray(t)
    dx = np.cos(t) + 4*np.cos(2*t)
    dy = -np.sin(t) + 4*np.sin(2*t)
    dz = -3*np.cos(3*t)
    return np.vstack([dx, dy, dz]).T

def integrand(s, t):
    # s и t - скаляры, возвращаем скаляр
    rs = r(s).flatten()
    rt = r(t).flatten()
    drs = dr(s).flatten()
    drt = dr(t).flatten()
    r12 = rs - rt
    cross = np.cross(drs, drt)
    numerator = np.dot(cross, r12)
    denominator = np.linalg.norm(r12)**3
    return numerator / denominator

# Вычисление интеграла зацепления
def link_integral():
    # 1/(4π) * double integral
    result, error = dblquad(integrand, 0, 2*np.pi, lambda t: 0, lambda t: 2*np.pi,
                            epsabs=1e-8, epsrel=1e-8)
    return result / (4*np.pi)

if __name__ == "__main__":
    print("Вычисляем интеграл зацепления для трилистника...")
    Lk = link_integral()
    print(f"Интеграл зацепления Lk = {Lk:.10f}")
    print(f"Ожидаемое значение (Wr + Tw) = 4")
    print(f"Отклонение: {abs(Lk - 4):.2e}")
