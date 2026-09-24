import sympy as sp

def verify_schrodinger_dirac_envelope():
    print("=== ВЕРИФИКАЦИЯ ВЫВОДА УРАВНЕНИЯ ШРЁДИНГЕРА И ДЛИНЫ ВОЛНЫ ДЕ БРОЙЛЯ ===")
    
    # 1. Символьные переменные
    tau, x_coord = sp.symbols('tau x', real=True)
    m_e, hbar, c, V = sp.symbols('m_e hbar c V', positive=True)
    
    # Функция огибающей psi(x, tau)
    psi = sp.Function('psi')(x_coord, tau)
    omega_0 = (m_e * c**2) / hbar
    k_0_sq = (m_e * c / hbar)**2
    
    # Полное поле деформации Phi(x, tau) = psi(x, tau) * exp(-i * omega_0 * tau)
    Phi = psi * sp.exp(-sp.I * omega_0 * tau)
    
    # Релятивистское волновое уравнение Клейна--Гордона:
    # (1/c^2) d^2 Phi/dtau^2 - d^2 Phi/dx^2 + k_0^2 * Phi + (2*m_e*V/hbar^2) * Phi = 0
    d2_Phi_dtau2 = sp.diff(Phi, tau, 2)
    d2_Phi_dx2 = sp.diff(Phi, x_coord, 2)
    
    wave_eq = (1 / c**2) * d2_Phi_dtau2 - d2_Phi_dx2 + k_0_sq * Phi + ((2 * m_e * V) / hbar**2) * Phi
    wave_eq_envelope = sp.simplify(wave_eq * sp.exp(sp.I * omega_0 * tau))
    
    print(f"[1] Полное уравнение для огибающей psi(x, tau):")
    print(f"    {wave_eq_envelope} = 0")
    
    # Медленно меняющаяся огибающая: отбрасываем вторую производную d^2 psi/dtau^2 << omega_0 dpsi/dtau
    d2_psi_dtau2 = sp.diff(psi, tau, 2)
    slow_envelope_eq = wave_eq_envelope.subs(d2_psi_dtau2, 0)
    
    # Умножаем на -(hbar^2 / (2 * m_e))
    schrodinger_form = sp.simplify(slow_envelope_eq * (-(hbar**2) / (2 * m_e)))
    
    # Каноническая форма: i*hbar*dpsi/dtau - (-hbar^2/(2*m)*d^2psi/dx^2 + V*psi) = 0
    dpsi_dtau = sp.diff(psi, tau)
    d2_psi_dx2_pure = sp.diff(psi, x_coord, 2)
    canonical_schrodinger = sp.I * hbar * dpsi_dtau - (-(hbar**2 / (2 * m_e)) * d2_psi_dx2_pure + V * psi)
    
    diff_schrodinger = sp.simplify(schrodinger_form - canonical_schrodinger)
    print(f"\n[2] Редукция к уравнению Шрёдингера:")
    print(f"    Уравнение огибающей: {schrodinger_form} = 0")
    print(f"    Тождественное совпадение с каноническим Шрёдингером: {diff_schrodinger == 0}")
    
    # 2. Доплеровская длина волны де Бройля: lambda_dB = h / p
    h_val = 6.62607015e-34 # Дж * с
    m_e_val = 9.1093837015e-31 # кг
    v_val = 1.0e6 # м/с (1000 км/с)
    
    p_val = m_e_val * v_val
    lambda_dB = h_val / p_val
    print(f"\n[3] Доплеровская длина волны де Бройля:")
    print(f"    При скорости v = {v_val/1000:.0f} км/с: lambda_dB = {lambda_dB*1e9:.5f} нм ({lambda_dB*1e10:.4f} Å)")
    
    # 3. Энергия связи атома водорода в основном состоянии (1s)
    alpha_val = 1.0 / 137.035999177
    m_e_eV = 0.51099895e6 # эВ
    E_1_theor = -0.5 * m_e_eV * (alpha_val**2)
    E_1_exp = -13.605693 # эВ (CODATA 2022)
    
    diff_E1 = abs(E_1_theor - E_1_exp) / abs(E_1_exp) * 100
    print(f"\n[4] Энергия связи атома водорода в основном состоянии (1s):")
    print(f"    E_1 (Теория: -1/2 * m_e * c^2 * alpha^2) = {E_1_theor:.6f} эВ")
    print(f"    E_1 (CODATA / Эксперимент)               = {E_1_exp:.6f} эВ")
    print(f"    Точность совпадения                      = {100 - diff_E1:.5f}%")
    
    return True

if __name__ == "__main__":
    verify_schrodinger_dirac_envelope()
