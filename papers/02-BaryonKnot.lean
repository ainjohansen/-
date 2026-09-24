-- ==============================================================================
-- ТОПОЛОГИЧЕСКАЯ ЭЛАСТОДИНАМИКА ВАКУУМА (ТЭВ)
-- Модуль: BaryonKnot.lean
-- Статья 02: Барионный трилистник T(3,2) и структура протона
-- Статус: Полная формальная верификация (0 допущений `sorry`)
-- ==============================================================================

import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic

namespace BaryonKnotModel

/-!
### 1. Функционал энергии узла на торе Клиффорда
-/

def knot_invariant (p q : ℕ) : ℚ :=
  (p^2 + q^2 : ℚ) + 2 / (p + q : ℚ)

lemma sq_ge_four {n : ℕ} (h : n ≥ 2) : n^2 ≥ 4 := by
  obtain ⟨k, hk⟩ := Nat.le.dest h
  subst hk
  have heq : (2 + k)^2 = k^2 + 4 * k + 4 := by ring
  rw [heq]
  omega

lemma sq_ge_sixteen {n : ℕ} (h : n ≥ 4) : n^2 ≥ 16 := by
  obtain ⟨k, hk⟩ := Nat.le.dest h
  subst hk
  have heq : (4 + k)^2 = k^2 + 8 * k + 16 := by ring
  rw [heq]
  omega

lemma sum_sq_ge_twenty (p q : ℕ) (hp : p ≥ 2) (hq : q ≥ 2) (hne : p ≠ q)
    (h32 : (p, q) ≠ (3, 2)) (h23 : (p, q) ≠ (2, 3)) : p^2 + q^2 ≥ 20 := by
  by_cases hq4 : q ≥ 4
  · have hp4 : p^2 ≥ 4 := sq_ge_four hp
    have hq16 : q^2 ≥ 16 := sq_ge_sixteen hq4
    omega
  · have hq_le : q ≤ 3 := by omega
    have hp4 : p ≥ 4 := by
      by_contra h_not
      have hp_le : p ≤ 3 := by omega
      have hp_cases : p = 2 ∨ p = 3 := by omega
      have hq_cases : q = 2 ∨ q = 3 := by omega
      rcases hp_cases with rfl | rfl <;> rcases hq_cases with rfl | rfl
      · exact False.elim (hne rfl)
      · exact False.elim (h23 rfl)
      · exact False.elim (h32 rfl)
      · exact False.elim (hne rfl)
    have hp16 : p^2 ≥ 16 := sq_ge_sixteen hp4
    have hq4 : q^2 ≥ 4 := sq_ge_four hq
    omega

/-- 
ТЕОРЕМА: Трилистник T(3,2) — единственный глобальный минимум спектра узлов.
-/
theorem trefoil_is_unique_ground_state :
    knot_invariant 3 2 = 67 / 5 ∧ 
    ∀ p q : ℕ, p ≥ 2 → q ≥ 2 → p ≠ q → (p, q) ≠ (3, 2) → (p, q) ≠ (2, 3) →
    knot_invariant p q > knot_invariant 3 2 := by
  constructor
  · unfold knot_invariant
    norm_num
  · intro p q hp hq hne h32 h23
    have h_sum_sq : p^2 + q^2 ≥ 20 := sum_sq_ge_twenty p q hp hq hne h32 h23
    have h_sum_sq_q : (p^2 + q^2 : ℚ) ≥ 20 := by norm_cast
    have h_pos : (2 : ℚ) / (p + q : ℚ) > 0 := by
      have : (p : ℚ) + (q : ℚ) > 0 := by positivity
      positivity
    have h_inv_gt_20 : knot_invariant p q > 20 := by
      unfold knot_invariant
      linarith
    have h_inv_32 : knot_invariant 3 2 = 67 / 5 := by
      unfold knot_invariant
      norm_num
    rw [h_inv_32]
    linarith

/-!
### 2. Форм-фактор отношения радиусов нуклона f = 3/4
-/

theorem nucleon_radii_ratio (lam_p : ℝ) (h_lam : lam_p > 0) :
    let r_c := 4 * lam_p
    let R_m := 3 * lam_p
    R_m / r_c = 3 / 4 := by
  intro r_c R_m
  dsimp [R_m, r_c]
  have h_ne : lam_p ≠ 0 := by linarith
  calc (3 * lam_p) / (4 * lam_p)
    _ = (3 / 4) * (lam_p / lam_p) := by ring
    _ = (3 / 4) * 1 := by rw [div_self h_ne]
    _ = 3 / 4 := by ring

/-!
### 3. Механизм Джулии--Зи: сохранение полного заряда пучностей
-/

theorem julia_zee_charge_conservation :
    let q₁ : ℚ := 2 / 3
    let q₂ : ℚ := 2 / 3
    let q₃ : ℚ := - 1 / 3
    q₁ + q₂ + q₃ = 1 := by
  intro q₁ q₂ q₃
  dsimp [q₁, q₂, q₃]
  norm_num

/-!
### 4. Строгая положительность отношения масс M_p / m_e
-/

noncomputable def mass_ratio_proton_electron (α : ℝ) : ℝ :=
  (67 / 5 : ℝ) * (1 / α) * (1 - (4 / 3 : ℝ) * α^2)

theorem mass_ratio_is_positive (α : ℝ) (h_α_pos : α > 0) (h_α_bound : α ≤ 1 / 2) :
    mass_ratio_proton_electron α > 0 := by
  dsimp [mass_ratio_proton_electron]
  have h_bracket : 1 - (4 / 3 : ℝ) * α^2 > 0 := by
    have h1 : α^2 ≤ (1 / 2 : ℝ)^2 := by nlinarith
    have h2 : (4 / 3 : ℝ) * α^2 ≤ (4 / 3 : ℝ) * (1 / 4 : ℝ) := by linarith
    linarith
  have h_factor : (67 / 5 : ℝ) * (1 / α) > 0 := by positivity
  positivity

end BaryonKnotModel