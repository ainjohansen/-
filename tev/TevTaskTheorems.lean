import tev.TevFoundations
import Mathlib.Tactic

/-!
# ЗАДАЧА: две теоремы топологической эластодинамики (дедукция из первых принципов)

Теорема A (Единственность основного состояния вакуума):
  Трилистник T(3,2) — единственный строгий минимум спектрального функционала
  Дирака--Лапласа I(p,q) = p² + q² + 2/(p+q) на множестве физически
  допустимых узлов (p ≥ 2, q ≥ 2, p > q, gcd(p,q) = 1).

Теорема B (Акустическая ОТО):
  Показатель преломления вакуума n = 1 + GN·M/(c²·r) строго больше единицы
  для любой положительной массы M (вакуум оптически плотнее у массы).

Обе теоремы выводятся дедуктивно из аксиом континуума, зафиксированных в
`tev.TevFoundations` (AdmissibleKnot, knotEnergy, vacuumRefractiveIndex).
-/

open TevFoundations

namespace TevTask

/-! ## Теорема A: единственность основного состояния вакуума -/

/-- Лемма: любой конкурирующий допустимый узел имеет сумму квадратов ≥ 25. -/
lemma competing_knots_base_ge_25 (p q : ℕ)
    (h : AdmissibleKnot p q) (hne : (p, q) ≠ (3, 2)) :
    25 ≤ p ^ 2 + q ^ 2 := by
  rcases h with ⟨hp, hq, _, hgt, hcop⟩
  by_cases hq3 : 3 ≤ q
  · -- q ≥ 3 и p > q ≥ 3  ⇒  p ≥ 4  ⇒  p² + q² ≥ 16 + 9 = 25
    have hp4 : 4 ≤ p := by omega
    nlinarith
  · -- q = 2 (т.к. q ≥ 2); p > 2, p ≠ 3 (из hne), p < 5  ⇒  p = 4
    have hq2 : q = 2 := by omega
    subst hq2
    by_cases hp5 : 5 ≤ p
    · nlinarith
    · interval_cases p
      · -- p = 2 противоречит p > q = 2
        contradiction
      · -- p = 4: gcd(4, 2) = 2 ≠ 1
        exfalso
        revert hcop
        decide

/-- ТЕОРЕМА A (Единственность основного состояния вакуума):
    Трилистник T(3,2) обладает строго наименьшей энергией среди ВСЕХ
    допустимых узлов. -/
theorem ground_state_uniqueness (p q : ℕ)
    (h : AdmissibleKnot p q) (hne : (p, q) ≠ (3, 2)) :
    knotEnergy 3 2 < knotEnergy p q := by
  -- Шаг 1: энергия трилистника тождественно равна 67/5 = 13.4
  have hE0 : knotEnergy 3 2 = 67 / 5 := by
    dsimp [knotEnergy]
    norm_num
  -- Шаг 2: любой конкурирующий допустимый узел имеет p² + q² ≥ 25
  have h25 := competing_knots_base_ge_25 p q h hne
  have h25Q : (25 : ℚ) ≤ (p ^ 2 + q ^ 2 : ℚ) := by exact_mod_cast h25
  -- Шаг 3: спектральная поправка 2/(p+q) строго положительна
  have hpos : (0 : ℚ) < 2 / (p + q : ℚ) := by
    rcases h with ⟨hp, hq, _⟩
    have : 0 < (p + q : ℚ) := by positivity
    positivity
  -- Шаг 4: 67/5 + 0 < 25 + 2/(p+q)
  rw [hE0]
  dsimp [knotEnergy]
  linarith

/-! ## Теорема B: акустическая ОТО (показатель преломления вакуума) -/

/-- ТЕОРЕМА B (Акустическая ОТО):
    Показатель преломления вакуума строго выше единицы
    (вакуум оптически плотнее у массы). -/
theorem acoustic_oto_index (GN M c r : ℝ)
    (hGN : 0 < GN) (hM : 0 < M) (hc : 0 < c) (hr : 0 < r) :
    1 < vacuumRefractiveIndex GN M c r := by
  dsimp [vacuumRefractiveIndex]
  have hc2_pos : 0 < c ^ 2 := sq_pos_of_pos hc
  have h_num : 0 < GN * M := mul_pos hGN hM
  have h_denom : 0 < c ^ 2 * r := mul_pos hc2_pos hr
  have h_frac : 0 < (GN * M) / (c ^ 2 * r) := div_pos h_num h_denom
  linarith

/-! ## Сверка с библиотекой: утверждения тождественны теоремам TevFoundations -/

/-- Теорема A совпадает с `TevFoundations.ground_state_is_strictly_trefoil`. -/
theorem ground_state_uniqueness_eq_library (p q : ℕ)
    (h : AdmissibleKnot p q) (hne : (p, q) ≠ (3, 2)) :
    knotEnergy 3 2 < knotEnergy p q :=
  TevFoundations.ground_state_is_strictly_trefoil p q h hne

/-- Теорема B совпадает с `TevFoundations.vacuum_index_gt_one`. -/
theorem acoustic_oto_index_eq_library (GN M c r : ℝ)
    (hGN : 0 < GN) (hM : 0 < M) (hc : 0 < c) (hr : 0 < r) :
    1 < vacuumRefractiveIndex GN M c r :=
  TevFoundations.vacuum_index_gt_one GN M c r hGN hM hc hr

end TevTask
