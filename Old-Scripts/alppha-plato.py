import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# ВХОДНЫЕ ДАННЫЕ (твои из BVP)
# ============================================================

I2 = 5.780647
I4 = 5.825971
I0 = 2.070358

# ============================================================
# ФУНКЦИИ
# ============================================================

def alpha_of_a(a):
    return (I4/a - I2*a) / (3 * I0 * a**3)

def energy(a):
    return a*I2 + (I4/a) + (a**3)*(alpha_of_a(a)*I0)

# численная производная
def d_alpha_da(a, h=1e-5):
    return (alpha_of_a(a+h) - alpha_of_a(a-h)) / (2*h)

# ============================================================
# СКАН
# ============================================================

a_vals = np.linspace(0.98, 1.02, 2000)

alpha_vals = np.array([alpha_of_a(a) for a in a_vals])
dalpha_vals = np.array([d_alpha_da(a) for a in a_vals])

# ============================================================
# ПОИСК ПЛАТО
# ============================================================

threshold = 1e-4  # критерий "почти плоско"

flat_indices = np.where(np.abs(dalpha_vals) < threshold)[0]

if len(flat_indices) > 0:
    a_min = a_vals[flat_indices[0]]
    a_max = a_vals[flat_indices[-1]]
    
    alpha_min = alpha_of_a(a_min)
    alpha_max = alpha_of_a(a_max)

    print("\n=== НАЙДЕНО ПЛАТО ===")
    print(f"a ∈ [{a_min:.6f}, {a_max:.6f}]")
    print(f"ширина по a: {(a_max - a_min)*100:.4f}%")
    print(f"α ∈ [{alpha_min:.6f}, {alpha_max:.6f}]")
    print(f"ширина по α: {(alpha_max - alpha_min)/alpha_min*100:.4f}%")
else:
    print("\n❌ ПЛАТО НЕ НАЙДЕНО")

# ============================================================
# ГРАФИКИ
# ============================================================

plt.figure(figsize=(10,5))

# alpha(a)
plt.subplot(1,2,1)
plt.plot(a_vals, alpha_vals)
plt.axhline(0.007297, linestyle='--')
plt.title("alpha(a)")
plt.xlabel("a")
plt.ylabel("alpha")

# производная
plt.subplot(1,2,2)
plt.plot(a_vals, dalpha_vals)
plt.axhline(0, linestyle='--')
plt.title("d alpha / da")
plt.xlabel("a")

plt.tight_layout()
plt.show()
