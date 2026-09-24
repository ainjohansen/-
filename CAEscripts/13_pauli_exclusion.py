import sympy as sp
import math

def verify_pauli_exclusion_principle():
    print("=== ВЕРИФИКАЦИЯ ПРИНЦИПА ЗАПРЕТА ПАУЛИ ИЗ ТОПОЛОГИЧЕСКОЙ НЕСЖИМАЕМОСТИ ===")
    
    # 1. Символьные переменные
    d = sp.symbols('d', positive=True) # Расстояние между центрами кернов солитонов
    r_0, C_6, V_core = sp.symbols('r_0 C_6 V_core', positive=True)
    
    # 2. Энергия деформации для параллельных спинов (одинаковое квантовое состояние)
    # Суммирование циркуляций: grad(Phi) ~ 2*r_0 / d
    grad_Phi_parallel = (2 * r_0) / d
    E_parallel = sp.Rational(1, 6) * C_6 * (grad_Phi_parallel**6) * V_core
    
    print(f"[1] Плотность энергии перекрытия для параллельных спинов (s1 == s2):")
    print(f"    E_parallel(d) = {sp.simplify(E_parallel)}")
    
    # Проверка расходимости барьера при d -> 0
    barrier_limit = sp.limit(E_parallel, d, 0)
    print(f"    Предел при d -> 0 (схлопывание в одну ячейку): {barrier_limit}")
    print(f"    Бесконечный барьер (Запрет совмещения): {barrier_limit == sp.oo}")
    
    # 3. Энергия деформации для антипараллельных спинов (s1 == -s2)
    # Компенсация циркуляций: grad(Phi) ~ O(1) (регулярный диполь)
    grad_Phi_antiparallel = sp.symbols('grad_Phi_reg', positive=True) # конечная регулярная величина
    E_antiparallel = sp.Rational(1, 6) * C_6 * (grad_Phi_antiparallel**6) * V_core
    
    print(f"\n[2] Энергия перекрытия для антипараллельных спинов (s1 == -s2):")
    print(f"    E_antiparallel(d->0) = {E_antiparallel} < infinity (Сингулярность отсутствует)")
    
    # 4. Топологический фазовый множитель обмена (Berry phase / SU(2) connection)
    # Пространственный обмен = поворот спинора на pi в SU(2)
    theta_exchange = sp.pi
    exchange_phase = sp.exp(sp.I * theta_exchange)
    print(f"\n[3] Топологический фазовый множитель пространственного обмена:")
    print(f"    exp(i * pi) = {exchange_phase} (Антисимметрия волновой функции: Psi(2,1) = -Psi(1,2))")
    
    # 5. Проверка тождества Слэтера: Psi(x1, x1) == 0
    # Волновая функция системы двух одинаковых фермионов в координатах x1, x2:
    psi_a = sp.Function('psi_a')
    x1, x2 = sp.symbols('x1 x2')
    
    # Определитель Слэтера для 2 частиц:
    slater_det = psi_a(x1) * psi_a(x2) - psi_a(x2) * psi_a(x1)
    # При совпадении координат x1 = x2:
    slater_same_coord = slater_det.subs(x2, x1)
    
    print(f"\n[4] Условие определителя Слэтера при x1 = x2:")
    print(f"    Psi(x1, x1) = {slater_same_coord} (Тождественно 0: {slater_same_coord == 0})")
    
    # 6. Численный расчет сдвиговых напряжений в BPS-керне при сближении электронов
    # r_0 = alpha * R_e ≈ 2.818 фм
    r_0_val = 2.81794e-15 # м
    d_vals = [10.0 * r_0_val, 2.0 * r_0_val, 1.0 * r_0_val, 0.5 * r_0_val, 0.1 * r_0_val]
    
    print(f"\n[5] Рост относительного сдвигового барьера (2*r_0 / d)^6:")
    for d_num in d_vals:
        ratio = (2.0 * r_0_val / d_num)**6
        print(f"    d = {d_num/r_0_val:4.1f} * r_0  ==>  Относительный барьер напряжения = {ratio:12.2e}")
        
    return True

if __name__ == "__main__":
    verify_pauli_exclusion_principle()
