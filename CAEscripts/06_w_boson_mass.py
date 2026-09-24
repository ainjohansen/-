import math
import sympy as sp

def verify_w_boson_mass():
    print("=== ВЕРИФИКАЦИЯ МАССЫ W-БОЗОНА И ЭЛЕКТРОСЛАБОГО СЕКТОРА ===")
    
    # 1. Физические константы
    alpha_inv = 137.035999177
    alpha = 1.0 / alpha_inv
    M_p_GeV = 0.938272088 # Масса протона в ГэВ
    
    print(f"[1] Базовые параметры:")
    print(f"    1/alpha = {alpha_inv:.6f}")
    print(f"    M_p = {M_p_GeV:.6f} ГэВ")
    
    # 2. Топологический фактор узла T(3,2)
    p = 3
    q = 2
    I_base = p**2 + q**2 # 13
    p_plus_q = p + q     # 5
    xi_W = math.sqrt(p_plus_q / I_base) # sqrt(5/13)
    
    print(f"\n[2] Топологический масштаб узла T(3,2):")
    print(f"    p + q = {p_plus_q}, p^2 + q^2 = {I_base}")
    print(f"    Геометрический фактор xi_W = sqrt(5/13) ≈ {xi_W:.7f}")
    
    # 3. Затравочная масса M_W^(0)
    E_yield = M_p_GeV / alpha
    M_W_0 = E_yield * xi_W
    print(f"\n[3] Затравочный масштаб предела текучести:")
    print(f"    E_yield = M_p / alpha = {E_yield:.4f} ГэВ")
    print(f"    M_W^(0) = E_yield * sqrt(5/13) = {M_W_0:.4f} ГэВ")
    
    # 4. Радиационная поправка пограничного слоя (7/2 * alpha/pi)
    pi_val = math.pi
    eps = alpha / pi_val
    rad_correction = 1.0 + 3.5 * eps
    
    M_W_theor = M_W_0 * rad_correction
    print(f"\n[4] Полная теоретическая масса W-бозона:")
    print(f"    Поправка пограничного слоя (1 + 7/2 * alpha/pi) = {rad_correction:.7f}")
    print(f"    M_W (Теория) = {M_W_theor:.4f} ГэВ ({M_W_theor*1000:.1f} МэВ)")
    
    # 5. Экспериментальное значение (PDG / World Average: 80.377 ± 0.012 ГэВ)
    M_W_exp = 80.377
    diff_pct = abs(M_W_theor - M_W_exp) / M_W_exp * 100
    print(f"\n[5] Сопоставление с экспериментом:")
    print(f"    M_W (PDG 2024 / World Average) = {M_W_exp:.4f} ГэВ")
    print(f"    Абсолютная погрешность = {abs(M_W_theor - M_W_exp)*1000:.1f} МэВ")
    print(f"    Точность вывода = {100 - diff_pct:.3f}% (Отклонение {diff_pct:.3f}%)")
    
    # 6. Угол Вайнберга sin^2(theta_W) = 3/13
    sin2_theta_W = 3.0 / 13.0
    sin2_theta_W_exp = 0.23122
    diff_sin2 = abs(sin2_theta_W - sin2_theta_W_exp) / sin2_theta_W_exp * 100
    print(f"\n[6] Угол Вайнберга:")
    print(f"    sin^2(theta_W) = 3/13 ≈ {sin2_theta_W:.6f}")
    print(f"    sin^2(theta_W) (PDG) = {sin2_theta_W_exp:.6f}")
    print(f"    Точность = {100 - diff_sin2:.2f}%")
    
    # 7. Константа Ферми G_F на электрослабом масштабе alpha(M_Z) ≈ 1/127.95
    alpha_MZ = 1.0 / 127.95 # Электрослабая поляризация пограничного слоя
    G_F_calc = (pi_val * alpha_MZ) / (math.sqrt(2) * (M_W_theor**2) * sin2_theta_W)
    G_F_exp = 1.1663788e-5 # ГэВ^-2
    diff_GF = abs(G_F_calc - G_F_exp) / G_F_exp * 100
    print(f"\n[7] Константа Ферми G_F (с учетом бегущей связности alpha(M_W)):")
    print(f"    G_F (Расчет) = {G_F_calc:.7e} ГэВ^-2")
    print(f"    G_F (PDG)    = {G_F_exp:.7e} ГэВ^-2")
    print(f"    Точность = {100 - diff_GF:.2f}% (Отклонение {diff_GF:.2f}%)")
    
    return True

if __name__ == "__main__":
    verify_w_boson_mass()
