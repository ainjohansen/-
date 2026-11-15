import Mathlib
import tev.TevFractionalLagrangian

set_option linter.style.whitespace false
set_option linter.style.emptyLine false
set_option linter.style.longLine false

/-!
# ТОПОЛОГИЧЕСКАЯ ЭЛАСТОДИНАМИКА ВАКУУМА (ТЭВ)
Модуль: TevFractionalIBP (Закрытие постулата `FractionalIBPDuality` по Фубини)

1. Интегрируемость и точный интеграл слабо-сингулярного ядра Вольтерра (t - τ)^(α - 1).
2. Дробное интегрирование по частям для интегралов Римана — Лиувилля
   на треугольнике {a < τ < t ≤ b} через `MeasureTheory.integrable_prod_iff`
   и `MeasureTheory.integral_integral_swap`:
     ∫_a^b p(t) · (I_{a+}^α q)(t) dt = ∫_a^b (I_{b-}^α p)(τ) · q(τ) dτ.
3. Вывод формулы Риве — Агравала для производной Капуто и правосторонней
   производной Римана — Лиувилля при η(a) = η(b) = 0 (полное снятие `FractionalIBPDuality`).
4. Безусловное дробное уравнение Эйлера — Лагранжа `riewe_euler_lagrange_unconditional`.
-/

namespace TevFractionalIBP

noncomputable section

open MeasureTheory

-- ============================================================================
-- 1. Интегрируемость и точный интеграл слабо-сингулярного ядра (t - τ)^(α - 1)
-- ============================================================================

lemma rpow_sub_left_integrable (α a t : ℝ) (hα : 0 < α) :
    IntervalIntegrable (fun τ => (t - τ) ^ (α - 1)) volume a t := by
  have h_rpow : IntervalIntegrable (fun u : ℝ => u ^ (α - 1)) volume 0 (t - a) :=
    intervalIntegral.intervalIntegrable_rpow' (by linarith)
  have h_comp := h_rpow.comp_sub_left t
  simp only [sub_zero, sub_sub_cancel] at h_comp
  exact h_comp.symm

lemma rpow_sub_left_integral (α a t : ℝ) (hα : 0 < α) :
    ∫ τ in a..t, (t - τ) ^ (α - 1) = (t - a) ^ α / α := by
  rw [intervalIntegral.integral_comp_sub_left (fun u : ℝ => u ^ (α - 1)) t (a := a) (b := t)]
  rw [sub_self]
  rw [integral_rpow (Or.inl (by linarith : -1 < α - 1))]
  have h_exp : α - 1 + 1 = α := by ring
  rw [h_exp, Real.zero_rpow (ne_of_gt hα), sub_zero]

-- ============================================================================
-- 2. Вспомогательные тождества по внутренним интегралам на треугольнике
-- ============================================================================

private lemma rl_left_inner_eq (α a b t : ℝ) (p q : ℝ → ℝ)
    (hα : 0 < α) (ht : t ∈ Set.Ioc a b) :
    ∫ τ in Set.Ioc a b, (if τ < t then p t * (t - τ) ^ (α - 1) * q τ else 0) =
      Real.Gamma α * (p t * TevFractionalLagrangian.riemannLiouvilleLeft α a q t) := by
  have h_ind : (fun τ => if τ < t then p t * (t - τ) ^ (α - 1) * q τ else 0) =
      (Set.Iio t).indicator (fun τ => p t * (t - τ) ^ (α - 1) * q τ) := rfl
  rw [h_ind, MeasureTheory.setIntegral_indicator measurableSet_Iio]
  have h_inter : Set.Ioc a b ∩ Set.Iio t = Set.Ioo a t := by
    ext τ
    simp only [Set.mem_inter_iff, Set.mem_Ioc, Set.mem_Iio, Set.mem_Ioo]
    exact ⟨fun h => ⟨h.1.1, h.2⟩, fun h => ⟨⟨h.1, le_trans (le_of_lt h.2) ht.2⟩, h.2⟩⟩
  rw [h_inter, MeasureTheory.setIntegral_congr_set Ioo_ae_eq_Ioc]
  rw [← intervalIntegral.integral_of_le (le_of_lt ht.1)]
  simp_rw [mul_assoc (p t)]
  rw [intervalIntegral.integral_const_mul]
  dsimp [TevFractionalLagrangian.riemannLiouvilleLeft]
  have hΓ : Real.Gamma α ≠ 0 := ne_of_gt (Real.Gamma_pos_of_pos hα)
  calc
    p t * ∫ τ in a..t, (t - τ) ^ (α - 1) * q τ
      = (Real.Gamma α * (1 / Real.Gamma α)) * (p t * ∫ τ in a..t, (t - τ) ^ (α - 1) * q τ) := by
          rw [mul_one_div_cancel hΓ, one_mul]
    _ = Real.Gamma α * (p t * ((1 / Real.Gamma α) * ∫ τ in a..t, (t - τ) ^ (α - 1) * q τ)) := by
          ring

private lemma rl_right_inner_eq (α a b τ : ℝ) (p q : ℝ → ℝ)
    (hα : 0 < α) (hτ : τ ∈ Set.Ioc a b) :
    ∫ t in Set.Ioc a b, (if τ < t then p t * (t - τ) ^ (α - 1) * q τ else 0) =
      Real.Gamma α * (TevFractionalLagrangian.riemannLiouvilleRight α b p τ * q τ) := by
  have h_ind : (fun t => if τ < t then p t * (t - τ) ^ (α - 1) * q τ else 0) =
      (Set.Ioi τ).indicator (fun t => p t * (t - τ) ^ (α - 1) * q τ) := rfl
  rw [h_ind, MeasureTheory.setIntegral_indicator measurableSet_Ioi]
  have h_inter : Set.Ioc a b ∩ Set.Ioi τ = Set.Ioc τ b := by
    ext t
    simp only [Set.mem_inter_iff, Set.mem_Ioc, Set.mem_Ioi]
    exact ⟨fun h => ⟨h.2, h.1.2⟩, fun h => ⟨⟨lt_trans hτ.1 h.1, h.2⟩, h.1⟩⟩
  rw [h_inter, ← intervalIntegral.integral_of_le hτ.2]
  have h_rew : (fun t => p t * (t - τ) ^ (α - 1) * q τ) =
      fun t => ((t - τ) ^ (α - 1) * p t) * q τ := by
    ext t; ring
  rw [h_rew, intervalIntegral.integral_mul_const]
  dsimp [TevFractionalLagrangian.riemannLiouvilleRight]
  have hΓ : Real.Gamma α ≠ 0 := ne_of_gt (Real.Gamma_pos_of_pos hα)
  calc
    (∫ t in τ..b, (t - τ) ^ (α - 1) * p t) * q τ
      = (Real.Gamma α * (1 / Real.Gamma α)) * ((∫ t in τ..b, (t - τ) ^ (α - 1) * p t) * q τ) := by
          rw [mul_one_div_cancel hΓ, one_mul]
    _ = Real.Gamma α * (((1 / Real.Gamma α) * ∫ t in τ..b, (t - τ) ^ (α - 1) * p t) * q τ) := by
          ring

-- ============================================================================
-- 3. Главная теорема Фубини для дробных интегралов Римана — Лиувилля
-- ============================================================================

-- ТЕОРЕМА 1 (ДРОБНОЕ ИНТЕГРИРОВАНИЕ ПО ЧАСТЯМ ДЛЯ I_{a+}^α ↔ I_{b-}^α):
-- Доказана строго без постулатов для любых непрерывных функций на [a, b] и α > 0.
theorem rl_fractional_ibp (α a b : ℝ) (hα : 0 < α) (hab : a ≤ b)
    (p q : ℝ → ℝ)
    (hp_cont : ContinuousOn p (Set.Icc a b))
    (hq_cont : ContinuousOn q (Set.Icc a b)) :
    ∫ t in a..b, p t * TevFractionalLagrangian.riemannLiouvilleLeft α a q t =
      ∫ τ in a..b, TevFractionalLagrangian.riemannLiouvilleRight α b p τ * q τ := by
  let μ : Measure ℝ := volume.restrict (Set.Ioc a b)
  have : IsFiniteMeasure μ := ⟨by
    dsimp [μ]
    rw [Measure.restrict_apply_univ]
    exact measure_Ioc_lt_top⟩
  obtain ⟨Mp0, hMp0⟩ := isCompact_Icc.exists_bound_of_continuousOn hp_cont
  obtain ⟨Mq0, hMq0⟩ := isCompact_Icc.exists_bound_of_continuousOn hq_cont
  let Mp : ℝ := max Mp0 0
  let Mq : ℝ := max Mq0 0
  have hMp_nonneg : 0 ≤ Mp := le_max_right _ _
  have hMq_nonneg : 0 ≤ Mq := le_max_right _ _
  have hp_bd : ∀ t ∈ Set.Icc a b, ‖p t‖ ≤ Mp := fun t ht => le_trans (hMp0 t ht) (le_max_left _ _)
  have hq_bd : ∀ τ ∈ Set.Icc a b, ‖q τ‖ ≤ Mq := fun τ hτ => le_trans (hMq0 τ hτ) (le_max_left _ _)
  let F : ℝ → ℝ → ℝ := fun t τ => if τ < t then p t * (t - τ) ^ (α - 1) * q τ else 0
  have hp_meas : AEStronglyMeasurable p μ :=
    (hp_cont.mono Set.Ioc_subset_Icc_self).aestronglyMeasurable measurableSet_Ioc
  have hq_meas : AEStronglyMeasurable q μ :=
    (hq_cont.mono Set.Ioc_subset_Icc_self).aestronglyMeasurable measurableSet_Ioc
  have hk_meas : AEStronglyMeasurable (fun z : ℝ × ℝ => (z.1 - z.2) ^ (α - 1)) (μ.prod μ) :=
    ((measurable_fst.sub measurable_snd).pow_const (α - 1)).aestronglyMeasurable
  have hF_meas : AEStronglyMeasurable (Function.uncurry F) (μ.prod μ) := by
    have h_prod_meas := (hp_meas.comp_fst.mul hk_meas).mul hq_meas.comp_snd
    exact h_prod_meas.indicator (measurableSet_lt measurable_snd measurable_fst)
  have hK_int_of_mem : ∀ t ∈ Set.Ioc a b,
      Integrable (fun τ => if τ < t then (t - τ) ^ (α - 1) else 0) μ := by
    intro t ht
    have h_ind : (fun τ => if τ < t then (t - τ) ^ (α - 1) else 0) =
        (Set.Iio t).indicator (fun τ => (t - τ) ^ (α - 1)) := rfl
    rw [h_ind, integrable_indicator_iff measurableSet_Iio]
    dsimp [IntegrableOn, μ]
    rw [Measure.restrict_restrict measurableSet_Iio]
    have h_inter : Set.Iio t ∩ Set.Ioc a b = Set.Ioo a t := by
      ext τ
      simp only [Set.mem_inter_iff, Set.mem_Iio, Set.mem_Ioc, Set.mem_Ioo]
      exact ⟨fun h => ⟨h.2.1, h.1⟩, fun h => ⟨h.2, ⟨h.1, le_trans (le_of_lt h.2) ht.2⟩⟩⟩
    rw [h_inter, Measure.restrict_congr_set Ioo_ae_eq_Ioc]
    exact (rpow_sub_left_integrable α a t hα).1
  have hK_integral_val : ∀ t ∈ Set.Ioc a b,
      ∫ τ, (if τ < t then (t - τ) ^ (α - 1) else 0) ∂μ = (t - a) ^ α / α := by
    intro t ht
    have h_ind : (fun τ => if τ < t then (t - τ) ^ (α - 1) else 0) =
        (Set.Iio t).indicator (fun τ => (t - τ) ^ (α - 1)) := rfl
    dsimp [μ]
    rw [h_ind, MeasureTheory.setIntegral_indicator measurableSet_Iio]
    have h_inter : Set.Ioc a b ∩ Set.Iio t = Set.Ioo a t := by
      ext τ
      simp only [Set.mem_inter_iff, Set.mem_Ioc, Set.mem_Iio, Set.mem_Ioo]
      exact ⟨fun h => ⟨h.1.1, h.2⟩, fun h => ⟨⟨h.1, le_trans (le_of_lt h.2) ht.2⟩, h.2⟩⟩
    rw [h_inter, MeasureTheory.setIntegral_congr_set Ioo_ae_eq_Ioc,
        ← intervalIntegral.integral_of_le (le_of_lt ht.1)]
    exact rpow_sub_left_integral α a t hα
  have h_F_bound : ∀ t ∈ Set.Ioc a b, ∀ τ ∈ Set.Ioc a b,
      ‖F t τ‖ ≤ (Mp * Mq) * (if τ < t then (t - τ) ^ (α - 1) else 0) := by
    intro t ht τ hτ
    dsimp [F]
    by_cases hτt : τ < t
    · rw [if_pos hτt, if_pos hτt]
      have h_rpow_nonneg : 0 ≤ (t - τ) ^ (α - 1) := Real.rpow_nonneg (by linarith) _
      have hp_t : |p t| ≤ Mp := hp_bd t (Set.Ioc_subset_Icc_self ht)
      have hq_τ : |q τ| ≤ Mq := hq_bd τ (Set.Ioc_subset_Icc_self hτ)
      change |p t * (t - τ) ^ (α - 1) * q τ| ≤ Mp * Mq * (t - τ) ^ (α - 1)
      rw [abs_mul, abs_mul, abs_of_nonneg h_rpow_nonneg]
      calc
        |p t| * (t - τ) ^ (α - 1) * |q τ|
          ≤ Mp * (t - τ) ^ (α - 1) * Mq :=
              mul_le_mul (mul_le_mul_of_nonneg_right hp_t h_rpow_nonneg) hq_τ (abs_nonneg _)
                (mul_nonneg hMp_nonneg h_rpow_nonneg)
        _ = Mp * Mq * (t - τ) ^ (α - 1) := by ring
    · rw [if_neg hτt, if_neg hτt]
      rw [abs_zero, mul_zero]
  have h_inner_of_mem : ∀ t ∈ Set.Ioc a b, Integrable (fun τ => F t τ) μ := by
    intro t ht
    have h_dom : Integrable (fun τ => (Mp * Mq) * (if τ < t then (t - τ) ^ (α - 1) else 0)) μ :=
      (hK_int_of_mem t ht).const_mul (Mp * Mq)
    have ht_meas : AEStronglyMeasurable (fun τ => F t τ) μ := by
      have hk_t : AEStronglyMeasurable (fun τ => (t - τ) ^ (α - 1)) μ :=
        ((measurable_const.sub measurable_id).pow_const (α - 1)).aestronglyMeasurable
      exact ((aestronglyMeasurable_const.mul hk_t).mul hq_meas).indicator measurableSet_Iio
    refine Integrable.mono' h_dom ht_meas ?_
    filter_upwards [ae_restrict_mem measurableSet_Ioc] with τ hτ
    exact h_F_bound t ht τ hτ
  have h_inner_int : ∀ᵐ t ∂μ, Integrable (fun τ => F t τ) μ := by
    filter_upwards [ae_restrict_mem measurableSet_Ioc] with t ht
    exact h_inner_of_mem t ht
  have h_outer_int : Integrable (fun t => ∫ τ, ‖F t τ‖ ∂μ) μ := by
    let C : ℝ := (Mp * Mq) * ((b - a) ^ α / α)
    have h_meas_norm : AEStronglyMeasurable (fun t => ∫ τ, ‖F t τ‖ ∂μ) μ :=
      hF_meas.norm.integral_prod_right'
    refine Integrable.mono' (integrable_const C) h_meas_norm ?_
    filter_upwards [ae_restrict_mem measurableSet_Ioc] with t ht
    have h_nonneg : 0 ≤ ∫ τ, ‖F t τ‖ ∂μ := integral_nonneg (fun _ => norm_nonneg _)
    rw [Real.norm_of_nonneg h_nonneg]
    have h_dom : Integrable (fun τ => (Mp * Mq) * (if τ < t then (t - τ) ^ (α - 1) else 0)) μ :=
      (hK_int_of_mem t ht).const_mul (Mp * Mq)
    have h_le_dom : ∫ τ, ‖F t τ‖ ∂μ ≤
        ∫ τ, (Mp * Mq) * (if τ < t then (t - τ) ^ (α - 1) else 0) ∂μ := by
      refine integral_mono_ae (h_inner_of_mem t ht).norm h_dom ?_
      filter_upwards [ae_restrict_mem measurableSet_Ioc] with τ hτ
      exact h_F_bound t ht τ hτ
    have h_eval_dom :
        (∫ τ, (Mp * Mq) * (if τ < t then (t - τ) ^ (α - 1) else 0) ∂μ) =
          (Mp * Mq) * ((t - a) ^ α / α) := by
      rw [integral_const_mul, hK_integral_val t ht]
    have h_pow_mono : (t - a) ^ α / α ≤ (b - a) ^ α / α := by
      apply div_le_div_of_nonneg_right _ (le_of_lt hα)
      exact Real.rpow_le_rpow (by linarith [ht.1]) (by linarith [ht.2]) (le_of_lt hα)
    have h_mul_mono : (Mp * Mq) * ((t - a) ^ α / α) ≤ C :=
      mul_le_mul_of_nonneg_left h_pow_mono (mul_nonneg hMp_nonneg hMq_nonneg)
    linarith
  have hF_prod : Integrable (Function.uncurry F) (μ.prod μ) :=
    (integrable_prod_iff hF_meas).mpr ⟨h_inner_int, h_outer_int⟩
  have h_swap := MeasureTheory.integral_integral_swap hF_prod
  have h_lhs_eq : (∫ t, ∫ τ, F t τ ∂μ ∂μ) =
      Real.Gamma α * ∫ t in a..b, p t * TevFractionalLagrangian.riemannLiouvilleLeft α a q t := by
    rw [intervalIntegral.integral_of_le hab, ← integral_const_mul]
    refine setIntegral_congr_fun measurableSet_Ioc ?_
    intro t ht
    exact rl_left_inner_eq α a b t p q hα ht
  have h_rhs_eq : (∫ τ, ∫ t, F t τ ∂μ ∂μ) =
      Real.Gamma α * ∫ τ in a..b, TevFractionalLagrangian.riemannLiouvilleRight α b p τ * q τ := by
    rw [intervalIntegral.integral_of_le hab, ← integral_const_mul]
    refine setIntegral_congr_fun measurableSet_Ioc ?_
    intro τ hτ
    exact rl_right_inner_eq α a b τ p q hα hτ
  rw [h_lhs_eq, h_rhs_eq] at h_swap
  exact mul_left_cancel₀ (ne_of_gt (Real.Gamma_pos_of_pos hα)) h_swap

-- ============================================================================
-- 4. Закрытие `FractionalIBPDuality` и безусловный Эйлер — Лагранж Риве
-- ============================================================================

/-- Правосторонняя дробная производная Римана — Лиувилля порядка γ:
    ^{RL}D_{b-}^γ p(τ) = - d/dτ (I_{b-}^{1-γ} p)(τ) -/
def riemannLiouvilleDerivRight (γ b : ℝ) (p : ℝ → ℝ) (τ : ℝ) : ℝ :=
  - deriv (TevFractionalLagrangian.riemannLiouvilleRight (1 - γ) b p) τ

-- ТЕОРЕМА 2 (ПОЛНОЕ СНЯТИЕ ПОСТУЛАТА `FractionalIBPDuality`):
-- Для γ < 1 и гладкой тестовой вариации η с нулевыми граничными условиями η(a) = η(b) = 0
-- левосторонняя производная Капуто строго перебрасывается на правостороннюю производную
-- Римана — Лиувилля без каких-либо постулатов двойственности.
theorem caputo_left_ibp_with_rl_right (γ a b : ℝ)
    (hγ_lt : γ < 1) (hab : a ≤ b)
    (p η P' : ℝ → ℝ)
    (hp_cont : ContinuousOn p (Set.Icc a b))
    (hP_deriv : ∀ τ ∈ Set.uIcc a b,
      HasDerivAt (TevFractionalLagrangian.riemannLiouvilleRight (1 - γ) b p) (P' τ) τ)
    (hP'_int : IntervalIntegrable P' volume a b)
    (hη_diff : ∀ τ ∈ Set.uIcc a b, HasDerivAt η (deriv η τ) τ)
    (hη'_cont : ContinuousOn (deriv η) (Set.Icc a b))
    (ha : η a = 0) (hb : η b = 0) :
    ∫ t in a..b, p t * TevFractionalLagrangian.caputoLeft γ a η t =
      ∫ τ in a..b, riemannLiouvilleDerivRight γ b p τ * η τ := by
  have h1γ : 0 < 1 - γ := by linarith
  have h_rl := rl_fractional_ibp (1 - γ) a b h1γ hab p (deriv η) hp_cont hη'_cont
  change ∫ t in a..b, p t * TevFractionalLagrangian.caputoLeft γ a η t = _
  dsimp [TevFractionalLagrangian.caputoLeft]
  rw [h_rl]
  have hη'_int : IntervalIntegrable (deriv η) volume a b :=
    hη'_cont.intervalIntegrable_of_Icc hab
  have h_ibp := intervalIntegral.integral_mul_deriv_eq_deriv_mul
    hP_deriv hη_diff hP'_int hη'_int
  rw [ha, hb, mul_zero, mul_zero, sub_zero, zero_sub] at h_ibp
  have h_eq : ∫ τ in a..b, TevFractionalLagrangian.riemannLiouvilleRight (1 - γ) b p τ * deriv η τ =
      ∫ τ in a..b, (- P' τ) * η τ := by
    rw [h_ibp, ← intervalIntegral.integral_neg]
    congr 1
    ext τ
    ring
  rw [h_eq]
  refine intervalIntegral.integral_congr ?_
  intro τ hτ
  dsimp [riemannLiouvilleDerivRight]
  rw [(hP_deriv τ hτ).deriv]

-- ТЕОРЕМА 3 (БЕЗУСЛОВНОЕ УРАВНЕНИЕ ЭЙЛЕРА — ЛАГРАНЖА РИВЕ — АГРАВАЛА):
-- Стационарная траектория δS_frac[u; η] = 0 удовлетворяет дробному уравнению Эйлера — Лагранжа
-- с точным правосторонним оператором ^{RL}D_{b-}^γ без привлечения `FractionalIBPDuality`.
theorem riewe_euler_lagrange_unconditional (γ a b ρ₀ k_eff : ℝ)
    (hγ_lt : γ < 1) (hab : a ≤ b)
    (u η P' : ℝ → ℝ)
    (hu_cap_cont : ContinuousOn (TevFractionalLagrangian.caputoLeft γ a u) (Set.Icc a b))
    (hP_deriv : ∀ τ ∈ Set.uIcc a b,
      HasDerivAt (TevFractionalLagrangian.riemannLiouvilleRight (1 - γ) b
        (TevFractionalLagrangian.caputoLeft γ a u)) (P' τ) τ)
    (hP'_int : IntervalIntegrable P' volume a b)
    (hη_diff : ∀ τ ∈ Set.uIcc a b, HasDerivAt η (deriv η τ) τ)
    (hη'_cont : ContinuousOn (deriv η) (Set.Icc a b))
    (ha : η a = 0) (hb : η b = 0)
    (h_stat : TevFractionalLagrangian.rieweFirstVariation γ a b ρ₀ k_eff u η = 0)
    (h_int1 : IntervalIntegrable (fun t =>
      ρ₀ * TevFractionalLagrangian.caputoLeft γ a u t *
        TevFractionalLagrangian.caputoLeft γ a η t) volume a b)
    (h_int2 : IntervalIntegrable (fun t => k_eff * u t * η t) volume a b)
    (h_int_adj : IntervalIntegrable (fun τ =>
      ρ₀ * riemannLiouvilleDerivRight γ b
        (TevFractionalLagrangian.caputoLeft γ a u) τ * η τ) volume a b) :
    ∫ τ in a..b,
      (ρ₀ * riemannLiouvilleDerivRight γ b
        (TevFractionalLagrangian.caputoLeft γ a u) τ - k_eff * u τ) * η τ = 0 := by
  have h_split : TevFractionalLagrangian.rieweFirstVariation γ a b ρ₀ k_eff u η =
      (∫ t in a..b, ρ₀ * TevFractionalLagrangian.caputoLeft γ a u t *
        TevFractionalLagrangian.caputoLeft γ a η t) -
      (∫ t in a..b, k_eff * u t * η t) := by
    dsimp [TevFractionalLagrangian.rieweFirstVariation]
    exact intervalIntegral.integral_sub h_int1 h_int2
  have h_ibp := caputo_left_ibp_with_rl_right γ a b hγ_lt hab
    (TevFractionalLagrangian.caputoLeft γ a u) η P'
    hu_cap_cont hP_deriv hP'_int hη_diff hη'_cont ha hb
  have h_dual :
      (∫ t in a..b, ρ₀ * TevFractionalLagrangian.caputoLeft γ a u t *
        TevFractionalLagrangian.caputoLeft γ a η t) =
      (∫ τ in a..b, ρ₀ * riemannLiouvilleDerivRight γ b
        (TevFractionalLagrangian.caputoLeft γ a u) τ * η τ) := by
    have h_assoc1 :
        (fun t => ρ₀ * TevFractionalLagrangian.caputoLeft γ a u t *
          TevFractionalLagrangian.caputoLeft γ a η t) =
        (fun t => ρ₀ * (TevFractionalLagrangian.caputoLeft γ a u t *
          TevFractionalLagrangian.caputoLeft γ a η t)) := by ext t; ring
    have h_assoc2 :
        (fun τ => ρ₀ * riemannLiouvilleDerivRight γ b
          (TevFractionalLagrangian.caputoLeft γ a u) τ * η τ) =
        (fun τ => ρ₀ * (riemannLiouvilleDerivRight γ b
          (TevFractionalLagrangian.caputoLeft γ a u) τ * η τ)) := by ext τ; ring
    rw [h_assoc1, h_assoc2, intervalIntegral.integral_const_mul,
        intervalIntegral.integral_const_mul, h_ibp]
  have h_sub_int :
      (∫ τ in a..b, ρ₀ * riemannLiouvilleDerivRight γ b
        (TevFractionalLagrangian.caputoLeft γ a u) τ * η τ) -
      (∫ t in a..b, k_eff * u t * η t) =
      ∫ τ in a..b,
        (ρ₀ * riemannLiouvilleDerivRight γ b
          (TevFractionalLagrangian.caputoLeft γ a u) τ - k_eff * u τ) * η τ := by
    rw [← intervalIntegral.integral_sub h_int_adj h_int2]
    congr 1
    ext τ
    ring
  rw [← h_sub_int, ← h_dual, ← h_split]
  exact h_stat

end

end TevFractionalIBP
