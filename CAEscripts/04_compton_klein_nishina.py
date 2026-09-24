import sympy as sp

def verify_compton_klein_nishina():
    print("=== ВЕРИФИКАЦИЯ КОМПТОНОВСКОГО РАССЕЯНИЯ И ФОРМУЛЫ КЛЕЙНА--НИШИНЫ ===")
    
    # 1. Символьные переменные
    alpha, hbar, c, m_e, omega, theta = sp.symbols('alpha hbar c m_e omega theta', positive=True)
    x = sp.symbols('x', positive=True) # x = hbar*omega / (m_e*c^2) — безразмерная энергия фотона
    print(f"[1] Безразмерная энергия фотона x = hbar*omega / (m_e*c^2)")
    
    # 2. Формула Комптона: отношение частот omega_prime / omega
    omega_ratio = 1 / (1 + x * (1 - sp.cos(theta)))
    print(f"[2] Отношение рассеянной частоты к падающей: omega'/omega = {omega_ratio}")
    
    # Проверка сдвига длины волны Delta(lambda) = lambda' - lambda
    # lambda = 2*pi*c / omega => Delta(lambda) = 2*pi*c * (1/omega' - 1/omega)
    # Заменяем x обратно на hbar*omega/(m_e*c^2) для проверки размерного сдвига
    omega_ratio_explicit = omega_ratio.subs(x, (hbar * omega) / (m_e * c**2))
    delta_lambda = 2 * sp.pi * c * ((1 / (omega * omega_ratio_explicit)) - (1 / omega))
    delta_lambda_simplified = sp.simplify(delta_lambda)
    expected_delta_lambda = (2 * sp.pi * hbar) / (m_e * c) * (1 - sp.cos(theta))
    
    print(f"[3] Сдвиг длины волны Delta(lambda) = {delta_lambda_simplified}")
    print(f"    Точное совпадение с h/(m_e*c)*(1 - cos(theta)): {sp.simplify(delta_lambda_simplified - expected_delta_lambda) == 0}")
    
    # 3. Дифференциальное сечение Клейна--Нишины
    r_e = (alpha * hbar) / (m_e * c) # Классический радиус электрона
    
    # dsigma/dOmega = (1/2) * r_e^2 * (omega'/omega)^2 * [ (omega'/omega) + (omega/omega') - sin(theta)^2 ]
    dsigma_KN = sp.Rational(1, 2) * (r_e**2) * (omega_ratio**2) * (omega_ratio + (1 / omega_ratio) - sp.sin(theta)**2)
    print(f"\n[4] Дифференциальное сечение Клейна--Нишины:")
    print(f"    dsigma/dOmega = {sp.simplify(dsigma_KN)}")
    
    # 4. Проверка предела Томсона (x -> 0 / низкие энергии)
    dsigma_Thomson = sp.limit(dsigma_KN, x, 0)
    expected_Thomson = (r_e**2 / 2) * (1 + sp.cos(theta)**2)
    diff_Thomson = sp.simplify(sp.trigsimp(dsigma_Thomson) - expected_Thomson)
    print(f"\n[5] Низкоэнергетический предел Томсона (x -> 0):")
    print(f"    dsigma/dOmega -> {sp.trigsimp(dsigma_Thomson)}")
    print(f"    Тождественное совпадение с (r_e^2/2)*(1 + cos^2(theta)): {diff_Thomson == 0}")
    
    # 5. Полное сечение Томсона: интеграл по сфере dOmega = 2*pi*sin(theta)*dtheta
    sigma_Thomson_total = sp.integrate(2 * sp.pi * sp.sin(theta) * expected_Thomson, (theta, 0, sp.pi))
    expected_total = sp.Rational(8, 3) * sp.pi * r_e**2
    print(f"\n[6] Полное сечение Томсона (интеграл по сфере):")
    print(f"    sigma_total = {sigma_Thomson_total}")
    print(f"    Тождественное совпадение с (8*pi/3)*r_e^2: {sigma_Thomson_total == expected_total}")
    
    # 6. Численный расчет полного сечения Томсона в барнах (1 барн = 1e-28 м^2)
    alpha_val = 1 / 137.035999177
    hbar_val = 1.054571817e-34 # Дж*с
    c_val = 299792458 # м/с
    m_e_val = 9.1093837015e-31 # кг
    
    r_e_num = (alpha_val * hbar_val) / (m_e_val * c_val)
    sigma_Th_num = (8 * 3.141592653589793 / 3) * (r_e_num**2)
    sigma_Th_barn = sigma_Th_num * 1e28
    print(f"\n[7] Численная проверка сечения Томсона:")
    print(f"    r_e = {r_e_num*1e15:.5f} фм (CODATA: 2.81794 фм)")
    print(f"    sigma_Thomson = {sigma_Th_barn:.5f} барн (PDG: 0.66524 барн)")
    
    return True

if __name__ == "__main__":
    verify_compton_klein_nishina()
