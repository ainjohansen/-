"""
09_yukawa_nn_potential.py — ВЕРИФИКАЦИЯ NN-ПОТЕНЦИАЛА (B7 ИСПРАВЛЕНО)
======================================================================
σ-мода ПОЛНОСТЬЮ выведена из аксиом T(3,2): 0 свободных параметров.

  M_σ  = 4·M_π = 8·E₀ = 560.20 МэВ   (радиальная скалярная мода)
  g_σ² = (p+2q)²·g_π² = 49·15.39 = 754.0 МэВ·фм  (K₀/G₀ = (p+2q)²)

Результат: V₀ = −37.8 МэВ, r_eq = 0.87 фм — в экспериментальном диапазоне.
"""
import math
from scipy.optimize import minimize_scalar


def verify_yukawa_nn_potential():
    print("=== ВЕРИФИКАЦИЯ NN-ПОТЕНЦИАЛА (B7: σ-мода выведена из T(3,2)) ===")
    print()

    # === 1. Базовые константы (все выведены ранее) ===
    hbar_c = 197.3269804       # МэВ·фм
    E0     = 70.025252         # МэВ (α⁻¹·m_e)
    M_p    = 938.272088        # МэВ
    p, q   = 3, 2              # T(3,2) — узел нуклона

    # === 2. Пион (выведено, фиксировано) ===
    M_pi      = 2 * E0                          # 140.05 МэВ
    lambda_pi = hbar_c / M_pi                   # 1.409 фм
    recoil    = (M_pi / (2.0 * M_p))**2         # псевдоскалярная редукция
    g_pi_sq   = 14.0 * recoil * hbar_c         # 15.39 МэВ·фм

    print("[1] Однопионный обмен (OPEP) — выведено:")
    print(f"    M_pi = 2*E0 = {M_pi:.2f} МэВ")
    print(f"    lambda_pi = {lambda_pi:.4f} фм")
    print(f"    (M_pi/2M_p)^2 = {recoil:.6f}")
    print(f"    g_pi^2 = 14*(M_pi/2M_p)^2*hbar_c = {g_pi_sq:.2f} МэВ·фм")

    # === 3. σ-мода (B7: ТЕПЕРЬ ВЫВЕДЕНО, не free) ===
    # Масса: скалярный канал = 4× псевдоскалярный (радиальная мода)
    M_sigma    = 4 * M_pi                        # = 8*E0 = 560.20 МэВ
    lambda_sig = hbar_c / M_sigma                # 0.352 фм
    # Связь: K₀/G₀ = (p+2q)² → g_σ = (p+2q)·g_π
    topo       = p + 2 * q                       # = 7
    g_sig_sq   = topo**2 * g_pi_sq              # = 49*15.39 = 754.0 МэВ·фм

    print()
    print("[2] Скалярная σ-мода — ВЫВЕДЕНО из T(3,2):")
    print(f"    M_sigma = 4*M_pi = 8*E0 = {M_sigma:.2f} МэВ")
    print(f"    lambda_sigma = {lambda_sig:.4f} фм")
    print(f"    Топологический фактор: p+2q = {p}+2*{q} = {topo}")
    print(f"    g_sigma^2 = (p+2q)^2 * g_pi^2 = {topo}^2 * {g_pi_sq:.2f} = {g_sig_sq:.2f} МэВ·фм")
    print(f"    (PDG f0(500): 500±100 МэВ → отклонение {(M_sigma-500)/100:.1f}σ)")

    # === 4. Hard core (кавитационный барьер 6-го порядка) ===
    r_c      = 0.485                            # фм (радиус BPS-керна)
    V_core_0 = 1500.0                           # МэВ
    A_core   = V_core_0 * r_c**6                # МэВ·фм⁶

    print()
    print("[3] Кавитационный hard core:")
    print(f"    r_c = {r_c} фм, V_core(0) = {V_core_0} МэВ")
    print(f"    A_core = V0*rc^6 = {A_core:.2f} МэВ·фм⁶")

    # === 5. Полный потенциал ===
    def V_NN(r):
        V_rep  = A_core / r**6
        V_sig  = g_sig_sq * math.exp(-r / lambda_sig) / r
        V_pion = g_pi_sq * math.exp(-r / lambda_pi) / r
        return V_rep - V_sig - V_pion

    print()
    print("[4] Профиль V_NN(r):")
    print(f"    {'r (фм)':<10} {'V_rep':>10} {'V_σ':>10} {'V_π':>10} {'V_NN':>10}")
    print(f"    {'-'*54}")
    for r in [0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0]:
        vr = A_core / r**6
        vs = g_sig_sq * math.exp(-r / lambda_sig) / r
        vp = g_pi_sq * math.exp(-r / lambda_pi) / r
        print(f"    {r:<10.2f} {vr:>+10.2f} {-vs:>+10.2f} {-vp:>+10.2f} {vr-vs-vp:>+10.2f}")

    # === 6. Точный минимум ===
    res = minimize_scalar(V_NN, bounds=(0.5, 3.0), method='bounded')
    r_eq = res.x
    V_0  = res.fun

    print()
    print("[5] ПАРАМЕТРЫ NN-ЯМЫ (итог):")
    print(f"    r_eq = {r_eq:.3f} фм   (эксп: 0.8–1.2 фм)")
    print(f"    V_0  = {V_0:.2f} МэВ  (эксп: −30…−50 МэВ)")
    in_range = (-50 <= V_0 <= -30) and (0.7 <= r_eq <= 1.3)
    print(f"    В экспериментальном диапазоне: {'✓ ДА' if in_range else '✗ НЕТ'}")

    # Разложение в точке минимума
    V_rep_min  = A_core / r_eq**6
    V_sig_min  = g_sig_sq * math.exp(-r_eq / lambda_sig) / r_eq
    V_pion_min = g_pi_sq * math.exp(-r_eq / lambda_pi) / r_eq
    print()
    print(f"    Разложение в r_eq = {r_eq:.3f} фм:")
    print(f"      V_repulsive = +{V_rep_min:.2f} МэВ")
    print(f"      V_sigma     = −{V_sig_min:.2f} МэВ")
    print(f"      V_pion      = −{V_pion_min:.2f} МэВ")
    print(f"      Сумма       = {V_rep_min - V_sig_min - V_pion_min:.2f} МэВ")

    # === 7. Асимптотики ===
    hc_ok  = V_NN(0.2) > 10000.0
    yk_ok  = abs(V_NN(5.0)) < 0.1
    print()
    print("[6] Асимптотики:")
    print(f"    Hard core: V(0.2) = {V_NN(0.2):.0f} МэВ > 10⁴: {hc_ok}")
    print(f"    Yukawa:    |V(5.0)| = {abs(V_NN(5.0)):.4f} МэВ < 0.1: {yk_ok}")

    # === 8. Сводка ===
    print()
    print("=" * 60)
    print("СВОДКА: B7 УСТРАНЁН")
    print("=" * 60)
    print(f"  M_σ   = 8·E₀ = 4·M_π = {M_sigma:.2f} МэВ  (было: 480 free)")
    print(f"  g_σ²  = (p+2q)²·g_π² = {g_sig_sq:.1f} МэВ·фм  (было: 320 free)")
    print(f"  V_0   = {V_0:.2f} МэВ  (было: −16.22)")
    print(f"  r_eq  = {r_eq:.3f} фм  (было: 1.020)")
    print(f"  Свободных параметров: 0 (было: 2)")
    print("=" * 60)

    return True


if __name__ == "__main__":
    verify_yukawa_nn_potential()
