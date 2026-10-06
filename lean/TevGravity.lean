import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# ТОПОЛОГИЧЕСКАЯ ЭЛАСТОДИНАМИКА ВАКУУМА (ТЭВ)
Часть 5. Гравитация Эшелби и акустическая ОТО (замедление часов)
-/

namespace TevGravity

/-! =========================================================================
    1. НЕСЖИМАЕМОСТЬ И РАДИАЛЬНОЕ СМЕЩЕНИЕ ЭШЕЛБИ
    В квазинесжимаемой среде (div u = 0) радиальное поле смещения каверны
    объема ΔV имеет единственно возможный профиль: u_r(r) = ΔV / (4π r²).
   ========================================================================= -/

def sphericalFlux (ΔV : ℝ) (r : ℝ) (ur : ℝ) : Prop :=
  4 * Real.pi * r ^ 2 * ur = ΔV

-- ТЕОРЕМА 1: Профиль радиального смещения несжимаемой среды однозначно равен ΔV / (4π r²)
theorem eshelby_displacement_profile (ΔV r : ℝ) (hr : r ≠ 0) (ur : ℝ)
    (h_flux : sphericalFlux ΔV r ur) :
    ur = ΔV / (4 * Real.pi * r ^ 2) := by
  dsimp [sphericalFlux] at h_flux
  have hr2 : r ^ 2 ≠ 0 := pow_ne_zero 2 hr
  have hpi : Real.pi ≠ 0 := Real.pi_ne_zero
  have h_denom : 4 * Real.pi * r ^ 2 ≠ 0 := by
    apply mul_ne_zero
    · apply mul_ne_zero (by norm_num) hpi
    · exact hr2
  rw [← h_flux]
  rw [mul_div_cancel_left₀ ur h_denom]

/-! =========================================================================
    2. УПРУГАЯ ЭНЕРГИЯ ВЗАИМОДЕЙСТВИЯ КАВЕРН И СИЛА НЬЮТОНА
    U_int(r) = - k_elast * (ΔV₁ * ΔV₂) / r
    F(r)     = - k_elast * (ΔV₁ * ΔV₂) / r²
   ========================================================================= -/

noncomputable def eshelbyInteractionEnergy (k_elast ΔV₁ ΔV₂ r : ℝ) : ℝ :=
  -k_elast * (ΔV₁ * ΔV₂) / r

noncomputable def eshelbyForce (k_elast ΔV₁ ΔV₂ r : ℝ) : ℝ :=
  -(k_elast * (ΔV₁ * ΔV₂) / r ^ 2)

-- ТЕОРЕМА 2: Связь потенциальной энергии и силы: U(r) = r * F(r)
theorem newton_force_from_eshelby (k_elast ΔV₁ ΔV₂ r : ℝ) (hr : r ≠ 0) :
    eshelbyInteractionEnergy k_elast ΔV₁ ΔV₂ r =
      r * eshelbyForce k_elast ΔV₁ ΔV₂ r := by
  dsimp [eshelbyInteractionEnergy, eshelbyForce]
  have hr2 : r ^ 2 ≠ 0 := pow_ne_zero 2 hr
  field_simp

-- ТЕОРЕМА 3: Доказательство гравитационного притяжения (сила строго отрицательна: F < 0)
theorem gravity_is_always_attractive (k_elast ΔV₁ ΔV₂ r : ℝ)
    (hk : 0 < k_elast) (hV1 : 0 < ΔV₁) (hV2 : 0 < ΔV₂) (hr : 0 < r) :
    eshelbyForce k_elast ΔV₁ ΔV₂ r < 0 := by
  dsimp [eshelbyForce]
  have hr2_pos : 0 < r ^ 2 := sq_pos_of_pos hr
  have hV_pos : 0 < ΔV₁ * ΔV₂ := mul_pos hV1 hV2
  have h_num : 0 < k_elast * (ΔV₁ * ΔV₂) := mul_pos hk hV_pos
  have h_frac : 0 < k_elast * (ΔV₁ * ΔV₂) / r ^ 2 := div_pos h_num hr2_pos
  linarith

/-! =========================================================================
    3. РЕПРЕЗЕНТАЦИЯ МАССЫ И ВЫВОД ГРАВИТАЦИОННОЙ ПОСТОЯННОЙ G_N
    Масса M = ρ₀ * ΔV.
    Сила принимает канонический вид Ньютона: F = - G_N * M₁ * M₂ / r².
   ========================================================================= -/

noncomputable def newtonForce (GN M₁ M₂ r : ℝ) : ℝ :=
  -(GN * (M₁ * M₂) / r ^ 2)

-- ТЕОРЕМА 4: Тождество силы Эшелби и закона всемирного тяготения Ньютона
theorem newton_gravitational_constant_bridge
    (k_elast ρ₀ M₁ M₂ r : ℝ) (hρ : ρ₀ ≠ 0) (hr : r ≠ 0) :
    eshelbyForce k_elast (M₁ / ρ₀) (M₂ / ρ₀) r =
      newtonForce (k_elast / ρ₀ ^ 2) M₁ M₂ r := by
  dsimp [eshelbyForce, newtonForce]
  have hρ2 : ρ₀ ^ 2 ≠ 0 := pow_ne_zero 2 hρ
  have hr2 : r ^ 2 ≠ 0 := pow_ne_zero 2 hr
  field_simp

/-! =========================================================================
    4. АКУСТИЧЕСКАЯ ОТО: СМЕЩЕНИЕ ЧАСТОТЫ ЧАСОВ (ГРАВИТАЦИОННОЕ КРАСНОЕ СМЕЩЕНИЕ)
    Часы Zitterbewegung идут с частотой ω(r) = ω₀ * (1 - GN * M / (c² * r)).
    Чем ближе к центру дефекта (r₁ < r₂), тем медленнее идут эталонные часы.
   ========================================================================= -/

noncomputable def localClockFrequency (ω₀ GN M c r : ℝ) : ℝ :=
  ω₀ * (1 - (GN * M) / (c ^ 2 * r))

-- ТЕОРЕМА 5: Замедление хода эталонных часов в напряженном континууме
theorem gravitational_time_dilation_clock_slowing
    (ω₀ GN M c r₁ r₂ : ℝ)
    (hω : 0 < ω₀) (hGN : 0 < GN) (hM : 0 < M) (hc : 0 < c)
    (hr1_pos : 0 < r₁) (hr12 : r₁ < r₂)
    (_h_weak_field : (GN * M) / (c ^ 2 * r₁) < 1) :
    localClockFrequency ω₀ GN M c r₁ < localClockFrequency ω₀ GN M c r₂ := by
  dsimp [localClockFrequency]
  have hc2_pos : 0 < c ^ 2 := sq_pos_of_pos hc
  have h_num_pos : 0 < GN * M := mul_pos hGN hM
  have h_denom1_pos : 0 < c ^ 2 * r₁ := mul_pos hc2_pos hr1_pos
  have h_denom_lt : c ^ 2 * r₁ < c ^ 2 * r₂ := by
    nlinarith
  have h_frac_gt : (GN * M) / (c ^ 2 * r₂) < (GN * M) / (c ^ 2 * r₁) := by
    exact div_lt_div_of_pos_left h_num_pos h_denom1_pos h_denom_lt
  have h_factor_lt : 1 - (GN * M) / (c ^ 2 * r₁) < 1 - (GN * M) / (c ^ 2 * r₂) := by
    linarith
  exact mul_lt_mul_of_pos_left h_factor_lt hω

end TevGravity
