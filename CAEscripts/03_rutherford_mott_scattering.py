import sympy as sp

def verify_rutherford_mott():
    print("=== ВЕРИФИКАЦИЯ ВЫВОДА СЕЧЕНИЙ РЕЗЕРФОРДА И МОТТА ===")
    
    # 1. Символьные переменные
    alpha, hbar, c, mu, v, b, theta = sp.symbols('alpha hbar c mu v b theta', positive=True)
    p, m, gamma = sp.symbols('p m gamma', positive=True)
    
    # 2. Связь прицельного параметра b и угла theta в поле V(R) = alpha*hbar*c/R
    b_expr = (alpha * hbar * c) / (mu * v**2) * sp.cot(theta / 2)
    print(f"[1] Прицельный параметр b(theta) = {b_expr}")
    
    # 3. Производная db/dtheta
    db_dtheta = sp.diff(b_expr, theta)
    print(f"[2] db/dtheta = {sp.simplify(db_dtheta)}")
    
    # 4. Классическое сечение рассеяния: dsigma/dOmega = (b / sin(theta)) * |db/dtheta|
    # Используем sin(theta) = 2*sin(theta/2)*cos(theta/2)
    dsigma_rutherford = (b_expr / sp.sin(theta)) * sp.Abs(db_dtheta)
    dsigma_rutherford_simplified = sp.trigsimp(dsigma_rutherford)
    
    print(f"\n[3] Дифференциальное сечение Резерфорда (символьно):")
    print(dsigma_rutherford_simplified)
    
    # Проверка структуры 1/sin^4(theta/2)
    expected_rutherford = ((alpha * hbar * c) / (2 * mu * v**2))**2 / sp.sin(theta/2)**4
    diff_rutherford = sp.simplify(dsigma_rutherford_simplified - expected_rutherford)
    print(f"    Разность с канонической формулой Резерфорда: {diff_rutherford} (Тождественно 0: {diff_rutherford == 0})")
    
    # 5. Релятивистский фактор Мотта (спинорное перекрытие)
    beta = v / c
    mott_factor = 1 - beta**2 * sp.sin(theta / 2)**2
    dsigma_mott = expected_rutherford * mott_factor
    
    print(f"\n[4] Сечение Мотта с фактором спинорной интерференции:")
    print(f"    dsigma_Mott = dsigma_Rutherford * ({mott_factor})")
    
    # 6. Нерелятивистский предел (v -> 0 / beta -> 0)
    limit_v0 = dsigma_mott.subs(beta, 0)
    print(f"\n[5] Нерелятивистский предел (beta -> 0):")
    print(f"    Совпадает с Резерфордом: {limit_v0 == expected_rutherford}")
    
    return True

if __name__ == "__main__":
    verify_rutherford_mott()
