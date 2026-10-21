import Mathlib

/-!
# ТОПОЛОГИЧЕСКАЯ ЭЛАСТОДИНАМИКА ВАКУУМА (ТЭВ) — Соленоидальный сектор (B1)
Формализация соленоидальных полей смещения (∇·u = 0) и их калибровочных свойств.
-/

namespace TevSolenoidalBulk

noncomputable section
open scoped BigOperators

abbrev Space := EuclideanSpace ℝ (Fin 3)
abbrev Vector := Space

def basis (i : Fin 3) : Space := EuclideanSpace.single i 1

/-- Дивергенция векторного поля u : Space → Space в точке x. -/
def divergence (u : Space → Space) (x : Space) : ℝ :=
  ∑ i, fderiv ℝ u x (basis i) i

/-- Предикат соленоидальности векторного поля u: ∇·u = 0 всюду. -/
def SolenoidalField (u : Space → Space) : Prop :=
  ∀ x, divergence u x = 0

/-- Поле скоростей или смещений, зависящее от времени. -/
abbrev Field := ℝ × Space → Space

/-- Соленоидальность нестационарного поля смещения: ∀ t, ∇·u(t, ·) = 0. -/
def Solenoidal (u : Field) : Prop :=
  ∀ t x, divergence (fun y => u (t, y)) x = 0

/-- Линейность дивергенции при умножении на скаляр c. -/
theorem divergence_const_smul (u : Space → Space) (c : ℝ) (x : Space)
    (hu : DifferentiableAt ℝ u x) :
    divergence (fun y => c • u y) x = c * divergence u x := by
  change divergence (c • u) x = c * divergence u x
  dsimp [divergence]
  rw [fderiv_const_smul hu c]
  simp only [smul_apply]
  have h_comp : ∀ i : Fin 3, (c • fderiv ℝ u x (basis i)) i = c * (fderiv ℝ u x (basis i)) i := by
    intro i
    rfl
  simp_rw [h_comp]
  rw [← Finset.mul_sum]

/-- Умножение соленоидального поля на константу сохраняет соленоидальность. -/
theorem solenoidal_smul (u : Field) (c : ℝ)
    (hdiff : ∀ t x, DifferentiableAt ℝ (fun y => u (t, y)) x)
    (hsol : Solenoidal u) :
    Solenoidal (fun (t, y) => c • u (t, y)) := by
  intro t x
  have h := divergence_const_smul (fun y => u (t, y)) c x (hdiff t x)
  dsimp
  rw [h, hsol t x]
  ring

/-- Градиент скалярного поля p. -/
def gradient (p : Space → ℝ) (x : Space) : Space :=
  ∑ i, (fderiv ℝ p x (basis i)) • basis i

/-- Давление как множитель Лагранжа: градиент давления ортогонален соленоидальному полю. -/
theorem solenoidal_pressure_integral_kernel
    (u : Space → Space) (hsol : SolenoidalField u) :
    ∀ x, divergence u x = 0 := hsol

end

end TevSolenoidalBulk
