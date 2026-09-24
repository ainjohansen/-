import sympy as sp
import math

def verify_bell_nonlocality_chsh():
    print("=== ВЕРИФИКАЦИЯ НЕЛОКАЛЬНОСТИ БЕЛЛА, CHSH И ПРЕДЕЛА ЦИРЕЛЬСОНА (2*sqrt(2)) ===")
    
    # 1. Символьные переменные
    theta_a, theta_a_prime = sp.symbols('theta_a theta_a_prime', real=True)
    theta_b, theta_b_prime = sp.symbols('theta_b theta_b_prime', real=True)
    
    # Квантовый коррелятор проекций фазовой связности: E(a, b) = -cos(theta_a - theta_b)
    def E_corr(t_1, t_2):
        return -sp.cos(t_1 - t_2)
    
    # 2. Функционал CHSH: S = E(a, b) - E(a, b') + E(a', b) + E(a', b')
    S_expr = E_corr(theta_a, theta_b) - E_corr(theta_a, theta_b_prime) + E_corr(theta_a_prime, theta_b) + E_corr(theta_a_prime, theta_b_prime)
    print(f"[1] Аналитический функционал CHSH:")
    print(f"    S(a, a', b, b') = {S_expr}")
    
    # 3. Вычисление для оптимальной конфигурации Цирельсона на торе Клиффорда:
    # theta_a = 0, theta_a' = pi/2, theta_b = pi/4, theta_b' = 3*pi/4
    angles = {
        theta_a: 0,
        theta_a_prime: sp.pi / 2,
        theta_b: sp.pi / 4,
        theta_b_prime: 3 * sp.pi / 4
    }
    
    E_ab = E_corr(theta_a, theta_b).subs(angles)
    E_ab_prime = E_corr(theta_a, theta_b_prime).subs(angles)
    E_aprime_b = E_corr(theta_a_prime, theta_b).subs(angles)
    E_aprime_b_prime = E_corr(theta_a_prime, theta_b_prime).subs(angles)
    
    print(f"\n[2] Значения корреляторов в углах Цирельсона:")
    print(f"    E(a, b)   = -cos(-pi/4)  = {E_ab}")
    print(f"    E(a, b')  = -cos(-3pi/4) = {E_ab_prime}")
    print(f"    E(a', b)  = -cos(pi/4)   = {E_aprime_b}")
    print(f"    E(a', b') = -cos(-pi/4)  = {E_aprime_b_prime}")
    
    # Полный функционал S
    S_val = sp.Abs(E_ab - E_ab_prime + E_aprime_b + E_aprime_b_prime)
    S_simplified = sp.simplify(S_val)
    expected_tsirelson = 2 * sp.sqrt(2)
    
    diff_S = sp.simplify(S_simplified - expected_tsirelson)
    print(f"\n[3] Итоговое значение функционала CHSH:")
    print(f"    S_theor = {S_simplified} ≈ {float(S_simplified):.6f}")
    print(f"    Классический локальный предел Белла (S <= 2) нарушен: {float(S_simplified) > 2.0}")
    print(f"    Тождественное совпадение с пределом Цирельсона (2*sqrt(2)): {diff_S == 0}")
    
    # 4. Доказательство теоремы No-Signaling (невозможность сверхсветовой связи)
    # P(A, B | a, b) = 1/4 * [1 - A * B * cos(theta_a - theta_b)]
    A_var, B_var = sp.symbols('A B')
    delta_theta = theta_a - theta_b
    
    P_joint = sp.Rational(1, 4) * (1 - A_var * B_var * sp.cos(delta_theta))
    
    # Суммирование по всем исходам B in {+1, -1}
    P_marginal_A_plus = (P_joint.subs({A_var: 1, B_var: 1}) + P_joint.subs({A_var: 1, B_var: -1}))
    P_marginal_A_minus = (P_joint.subs({A_var: -1, B_var: 1}) + P_joint.subs({A_var: -1, B_var: -1}))
    
    print(f"\n[4] Маргинальные вероятности измерений на детекторе A:")
    print(f"    P(A = +1 | theta_a, theta_b) = {sp.simplify(P_marginal_A_plus)}")
    print(f"    P(A = -1 | theta_a, theta_b) = {sp.simplify(P_marginal_A_minus)}")
    print(f"    Зависимость от угла детектора B отсутствует (No-Signaling Theorem): {sp.diff(P_marginal_A_plus, theta_b) == 0}")
    
    return True

if __name__ == "__main__":
    verify_bell_nonlocality_chsh()
