import math

def verify_deuteron_properties():
    print("=== ВЕРИФИКАЦИЯ ЭНЕРГИИ СВЯЗИ И СВОЙСТВ ДЕЙТРОНА (2H) ===")
    
    # 1. Базовые константы из реестра
    hbar_c = 197.3269804 # МэВ * фм
    E0_MeV = 70.025252
    M_p_MeV = 938.272088
    alpha_inv = 137.035999177
    alpha = 1.0 / alpha_inv
    pi_val = math.pi
    
    # 2. Масса пиона и энергия связи E_b
    M_pi = 2.0 * E0_MeV # 140.0505 МэВ
    p = 3
    q = 2
    I_base = p**2 + q**2 # 13
    delta_N_tau = 14     # N_tau - 1
    
    E_scale = (M_pi**2) / (2.0 * M_p_MeV)
    E_b_theor = E_scale * (p / delta_N_tau)
    E_b_exp = 2.224566 # МэВ
    
    diff_Eb = abs(E_b_theor - E_b_exp) / E_b_exp * 100
    print(f"[1] Энергия связи дейтрона E_b:")
    print(f"    E_b (Теория: M_pi^2/(2*M_p) * 3/14) = {E_b_theor:.5f} МэВ ({E_b_theor*1000:.1f} кэВ)")
    print(f"    E_b (CODATA / Эксперимент)         = {E_b_exp:.5f} МэВ ({E_b_exp*1000:.1f} кэВ)")
    print(f"    Точность вывода = {100 - diff_Eb:.3f}% (Отклонение {diff_Eb:.3f}%)")
    
    # 3. Параметры квантового гало дейтрона
    kappa = math.sqrt(M_p_MeV * E_b_exp) / hbar_c # фм^-1 (0.23153 фм^-1)
    lambda_pi = hbar_c / M_pi # 1.40897 фм
    print(f"\n[2] Параметры волнового квантового гало:")
    print(f"    Волновой параметр kappa = {kappa:.5f} фм^-1")
    print(f"    Длина квантового гало 1/kappa = {1.0/kappa:.3f} фм")
    
    # 4. Доля D-волны (L=2) из угла Вайнберга: P_D = (1/4) * sin^2(theta_W) = 3/52
    sin2_theta_W = p / I_base # 3/13
    P_D_theor = 0.25 * sin2_theta_W # 3/52 ≈ 5.769%
    P_D_exp = 0.0576 # Эксперимент (Bonn potential: 5.76%)
    
    print(f"\n[3] Доля D-волнового тензорного состояния P_D:")
    print(f"    P_D (Теория: 1/4 * sin^2(theta_W) = 3/52) = {P_D_theor*100:.3f}%")
    print(f"    P_D (Bonn potential / Эксперимент)        = {P_D_exp*100:.2f}%")
    print(f"    Точность совпадения = {100 - abs(P_D_theor - P_D_exp)/P_D_exp*100:.2f}%")
    
    # 5. Магнитный момент дейтрона mu_d с учетом обменных токов (MEC)
    mu_p = +2.79284735
    mu_n = -1.91304273
    mu_sum = mu_p + mu_n
    
    # Вклад мезонных обменных токов (MEC): delta_mu_MEC = (alpha/pi) * (13/3) mu_N
    delta_mu_MEC = (alpha / pi_val) * (I_base / p) # ≈ +0.010065 mu_N
    mu_d_theor = (mu_sum - 1.5 * (mu_sum - 0.5) * P_D_theor) + delta_mu_MEC
    mu_d_exp = 0.85743823
    
    diff_mu = abs(mu_d_theor - mu_d_exp) / mu_d_exp * 100
    print(f"\n[4] Магнитный момент дейтрона mu_d:")
    print(f"    Вклад обменных токов MEC   = +{delta_mu_MEC:.6f} mu_N")
    print(f"    mu_d (Теория)              = {mu_d_theor:.6f} mu_N")
    print(f"    mu_d (CODATA / Эксперимент) = {mu_d_exp:.6f} mu_N")
    print(f"    Точность вывода            = {100 - diff_mu:.3f}% (Отклонение {diff_mu:.3f}%)")
    
    # 6. Электрический квадрупольный момент Q_d = (1 / (16 * kappa^2)) * sin^2(theta_W) * (1 + delta_core)
    Q_d_0 = (1.0 / (16.0 * (kappa**2))) * sin2_theta_W
    delta_core = kappa * lambda_pi * ((p - 1.0) / (p + q)) # 0.23153 * 1.409 * (2/5) ≈ 0.0626 (6.26%)
    Q_d_theor = Q_d_0 * (1.0 + delta_core)
    Q_d_exp = 0.2859 # e * фм^2
    
    diff_Qd = abs(Q_d_theor - Q_d_exp) / Q_d_exp * 100
    print(f"\n[5] Электрический квадрупольный момент Q_d:")
    print(f"    Затравочный момент Q_d^(0) = 1/(16*kappa^2) * sin^2(theta_W) = {Q_d_0:.4f} e * фм^2")
    print(f"    Поправка конечного размера BPS-керна (1 + delta_core)        = {1.0 + delta_core:.4f}")
    print(f"    Q_d (Теория)                                                 = {Q_d_theor:.4f} e * фм^2")
    print(f"    Q_d (CODATA / Эксперимент)                                   = {Q_d_exp:.4f} e * фм^2")
    print(f"    Точность вывода                                              = {100 - diff_Qd:.2f}% (Отклонение {diff_Qd:.2f}%)")
    
    return True

if __name__ == "__main__":
    verify_deuteron_properties()
