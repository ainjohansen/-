import Mathlib
import tev.TevSolenoidalBulk
import tev.TevNambuMises

set_option linter.style.whitespace false
set_option linter.style.emptyLine false
set_option linter.style.longLine false

/-!
# ТОПОЛОГИЧЕСКАЯ ЭЛАСТОДИНАМИКА ВАКУУМА (ТЭВ)
Модуль: TevNambuSchwarzC2 (Настоящий C²-анализ и теорема Шварца без джетов)

1. Вывод коммутативности вторых частных производных ∂_j ∂_i f = ∂_i ∂_j f
   для реальных функций класса `ContDiff ℝ 2` через `ContDiffAt.isSymmSndFDerivAt`.
2. Доказательство строгой соленоидальности потока Намбу ∇ · (∇H₁ × ∇H₂) ≡ 0
   и ортогональности градиентам для настоящих C²-потенциалов H₁, H₂ : Space → ℝ.
3. Прямая сшивка с `TevNambuMises`: C²-гладкость H₁, H₂ вынуждает изохорность
   деформации и независимость диссипации Мизеса от гидростатического давления.
4. Бонусное закрытие `TevMaxwell`: ∇ · (∇ × u) ≡ 0 для реальных C²-полей смещений u : Space → Space.
-/

namespace TevNambuSchwarzC2

noncomputable section

open scoped BigOperators RealInnerProductSpace

abbrev Space := TevSolenoidalBulk.Space
abbrev e (i : Fin 3) : Space := TevSolenoidalBulk.basis i

/-- Первая частная производная скалярного поля f : Space → ℝ вдоль базисного вектора e_i -/
def partial1 (f : Space → ℝ) (i : Fin 3) : Space → ℝ :=
  fun x => fderiv ℝ f x (e i)

lemma space_add_apply (u v : Space) (i : Fin 3) : (u + v) i = u i + v i := rfl
lemma space_smul_apply (c : ℝ) (u : Space) (i : Fin 3) : (c • u) i = c * u i := rfl

lemma space_sum3_apply (g : Fin 3 → Space) (j : Fin 3) :
    (∑ i : Fin 3, g i) j = g 0 j + g 1 j + g 2 j := by
  rw [Fin.sum_univ_three]
  rfl

lemma clm_add_apply (A B : Space →L[ℝ] Space) (v : Space) : (A + B) v = A v + B v := rfl
lemma clm_smulRight_apply (L : Space →L[ℝ] ℝ) (u v : Space) :
    (L.smulRight u) v = L v • u := rfl

lemma e0_0 : (TevSolenoidalBulk.basis 0) 0 = 1 := by
  simp [TevSolenoidalBulk.basis, EuclideanSpace.single]
lemma e0_1 : (TevSolenoidalBulk.basis 0) 1 = 0 := by
  simp [TevSolenoidalBulk.basis, EuclideanSpace.single]
lemma e0_2 : (TevSolenoidalBulk.basis 0) 2 = 0 := by
  simp [TevSolenoidalBulk.basis, EuclideanSpace.single]
lemma e1_0 : (TevSolenoidalBulk.basis 1) 0 = 0 := by
  simp [TevSolenoidalBulk.basis, EuclideanSpace.single]
lemma e1_1 : (TevSolenoidalBulk.basis 1) 1 = 1 := by
  simp [TevSolenoidalBulk.basis, EuclideanSpace.single]
lemma e1_2 : (TevSolenoidalBulk.basis 1) 2 = 0 := by
  simp [TevSolenoidalBulk.basis, EuclideanSpace.single]
lemma e2_0 : (TevSolenoidalBulk.basis 2) 0 = 0 := by
  simp [TevSolenoidalBulk.basis, EuclideanSpace.single]
lemma e2_1 : (TevSolenoidalBulk.basis 2) 1 = 0 := by
  simp [TevSolenoidalBulk.basis, EuclideanSpace.single]
lemma e2_2 : (TevSolenoidalBulk.basis 2) 2 = 1 := by
  simp [TevSolenoidalBulk.basis, EuclideanSpace.single]

-- ============================================================================
-- 1. Теорема Шварца для реальных C²-функций на ℝ³
-- ============================================================================

lemma c2_fderiv_differentiable (f : Space → ℝ) (hf : ContDiff ℝ 2 f) :
    Differentiable ℝ (fderiv ℝ f) := by
  have h1 : ContDiff ℝ 1 (fderiv ℝ f) :=
    ((contDiff_succ_iff_fderiv (n := 1)).mp (by rw [one_add_one_eq_two]; exact hf)).2.2
  exact h1.differentiable one_ne_zero

lemma partial1_differentiableAt (f : Space → ℝ) (hf : ContDiff ℝ 2 f) (i : Fin 3) (x : Space) :
    DifferentiableAt ℝ (partial1 f i) x := by
  have hdf := (c2_fderiv_differentiable f hf) x
  change DifferentiableAt ℝ (fun y => fderiv ℝ f y (e i)) x
  exact (hdf.hasFDerivAt.clm_apply (hasFDerivAt_const (e i) x)).differentiableAt

lemma partial1_fderiv (f : Space → ℝ) (hf : ContDiff ℝ 2 f) (i j : Fin 3) (x : Space) :
    partial1 (partial1 f i) j x = fderiv ℝ (fderiv ℝ f) x (e j) (e i) := by
  have hdf := (c2_fderiv_differentiable f hf) x
  change fderiv ℝ (fun y => fderiv ℝ f y (e i)) x (e j) = _
  rw [fderiv_clm_apply hdf (differentiableAt_const (e i))]
  simp [ContinuousLinearMap.comp_zero, ContinuousLinearMap.flip_apply]

-- ТЕОРЕМА 1 (НАСТОЯЩАЯ ТЕОРЕМА ШВАРЦА ДЛЯ C²-ПОЛЕЙ):
-- Для любой функции f класса ContDiff ℝ 2 на ℝ³ смешанные производные коммутируют
theorem schwarz_c2_comm (f : Space → ℝ) (hf : ContDiff ℝ 2 f) (i j : Fin 3) (x : Space) :
    partial1 (partial1 f i) j x = partial1 (partial1 f j) i x := by
  rw [partial1_fderiv f hf i j x, partial1_fderiv f hf j i x]
  exact hf.contDiffAt.isSymmSndFDerivAt
    (by rw [minSmoothness_of_isRCLikeNormedField]) (e j) (e i)

-- ============================================================================
-- 2. Поток Намбу для C²-потенциалов H₁, H₂ : ортогональность и соленоидальность
-- ============================================================================

/-- Поток Намбу v = ∇H₁ × ∇H₂ для двух скалярных полей H₁, H₂ : Space → ℝ -/
def nambuVectorField (H₁ H₂ : Space → ℝ) (x : Space) : Space :=
  (partial1 H₁ 1 x * partial1 H₂ 2 x - partial1 H₁ 2 x * partial1 H₂ 1 x) • e 0 +
  (partial1 H₁ 2 x * partial1 H₂ 0 x - partial1 H₁ 0 x * partial1 H₂ 2 x) • e 1 +
  (partial1 H₁ 0 x * partial1 H₂ 1 x - partial1 H₁ 1 x * partial1 H₂ 0 x) • e 2

lemma space_inner_basis_sum (u v : Space) :
    inner ℝ u v = u 0 * v 0 + u 1 * v 1 + u 2 * v 2 := by
  rw [PiLp.inner_apply, Fin.sum_univ_three]
  simp only [RCLike.inner_apply, conj_trivial]
  ring

-- ТЕОРЕМА 2A: Поток Намбу строго ортогонален градиенту геометрии ∇H₁
theorem nambu_c2_orthogonal_H1 (H₁ H₂ : Space → ℝ) (x : Space) :
    inner ℝ (TevSolenoidalBulk.gradient H₁ x) (nambuVectorField H₁ H₂ x) = 0 := by
  rw [space_inner_basis_sum]
  dsimp [TevSolenoidalBulk.gradient, nambuVectorField, partial1, e]
  simp only [space_sum3_apply, space_smul_apply,
    e0_0, e0_1, e0_2, e1_0, e1_1, e1_2, e2_0, e2_1, e2_2]
  ring

-- ТЕОРЕМА 2B: Поток Намбу строго ортогонален градиенту топологии ∇H₂
theorem nambu_c2_orthogonal_H2 (H₁ H₂ : Space → ℝ) (x : Space) :
    inner ℝ (TevSolenoidalBulk.gradient H₂ x) (nambuVectorField H₁ H₂ x) = 0 := by
  rw [space_inner_basis_sum]
  dsimp [TevSolenoidalBulk.gradient, nambuVectorField, partial1, e]
  simp only [space_sum3_apply, space_smul_apply,
    e0_0, e0_1, e0_2, e1_0, e1_1, e1_2, e2_0, e2_1, e2_2]
  ring

lemma fderiv_mul_sub_apply (f₁ f₂ g₁ g₂ : Space → ℝ) (x v : Space)
    (hf₁ : DifferentiableAt ℝ f₁ x) (hf₂ : DifferentiableAt ℝ f₂ x)
    (hg₁ : DifferentiableAt ℝ g₁ x) (hg₂ : DifferentiableAt ℝ g₂ x) :
    fderiv ℝ (fun y => f₁ y * g₁ y - f₂ y * g₂ y) x v =
      (fderiv ℝ f₁ x v * g₁ x + f₁ x * fderiv ℝ g₁ x v) -
      (fderiv ℝ f₂ x v * g₂ x + f₂ x * fderiv ℝ g₂ x v) := by
  have hm₁ := hf₁.hasFDerivAt.mul hg₁.hasFDerivAt
  have hm₂ := hf₂.hasFDerivAt.mul hg₂.hasFDerivAt
  have h_diff : HasFDerivAt (fun y => f₁ y * g₁ y - f₂ y * g₂ y)
      ((f₁ x • fderiv ℝ g₁ x + g₁ x • fderiv ℝ f₁ x) -
       (f₂ x • fderiv ℝ g₂ x + g₂ x • fderiv ℝ f₂ x)) x :=
    hm₁.sub hm₂
  have h_rfl :
      ((f₁ x • fderiv ℝ g₁ x + g₁ x • fderiv ℝ f₁ x) -
       (f₂ x • fderiv ℝ g₂ x + g₂ x • fderiv ℝ f₂ x)) v =
      (f₁ x * fderiv ℝ g₁ x v + g₁ x * fderiv ℝ f₁ x v) -
      (f₂ x * fderiv ℝ g₂ x v + g₂ x * fderiv ℝ f₂ x v) := by
    change (f₁ x • fderiv ℝ g₁ x v + g₁ x • fderiv ℝ f₁ x v) -
           (f₂ x • fderiv ℝ g₂ x v + g₂ x • fderiv ℝ f₂ x v) = _
    simp only [smul_eq_mul]
  rw [h_diff.fderiv, h_rfl]
  ring

lemma divergence_three_basis_sum (a₀ a₁ a₂ : Space → ℝ) (x : Space)
    (ha₀ : DifferentiableAt ℝ a₀ x)
    (ha₁ : DifferentiableAt ℝ a₁ x)
    (ha₂ : DifferentiableAt ℝ a₂ x) :
    TevSolenoidalBulk.divergence
      (fun y => a₀ y • e 0 + a₁ y • e 1 + a₂ y • e 2) x =
      fderiv ℝ a₀ x (e 0) + fderiv ℝ a₁ x (e 1) + fderiv ℝ a₂ x (e 2) := by
  have hd : HasFDerivAt (fun y => a₀ y • e 0 + a₁ y • e 1 + a₂ y • e 2)
      ((fderiv ℝ a₀ x).smulRight (e 0) + (fderiv ℝ a₁ x).smulRight (e 1) +
       (fderiv ℝ a₂ x).smulRight (e 2)) x :=
    ((ha₀.hasFDerivAt.smul_const (e 0)).add
      (ha₁.hasFDerivAt.smul_const (e 1))).add (ha₂.hasFDerivAt.smul_const (e 2))
  dsimp [TevSolenoidalBulk.divergence]
  rw [Fin.sum_univ_three, hd.fderiv]
  dsimp [e]
  simp only [clm_add_apply, clm_smulRight_apply, space_add_apply, space_smul_apply,
    e0_0, e0_1, e0_2, e1_0, e1_1, e1_2, e2_0, e2_1, e2_2]
  ring

-- ТЕОРЕМА 3 (ГЛАВНАЯ — СОЛЕНОИДАЛЬНОСТЬ ПОТОКА НАМБУ ДЛЯ C²-ПОТЕНЦИАЛОВ):
-- Для любых дважды непрерывно дифференцируемых полей H₁, H₂ : Space → ℝ
-- поток Намбу ∇H₁ × ∇H₂ тождественно соленоидален (∇ · v ≡ 0 всюду в ℝ³)
-- БЕЗ каких-либо ручных допущений о симметрии производных.
theorem nambu_c2_solenoidal (H₁ H₂ : Space → ℝ)
    (h1 : ContDiff ℝ 2 H₁) (h2 : ContDiff ℝ 2 H₂) :
    TevSolenoidalBulk.SolenoidalField (nambuVectorField H₁ H₂) := by
  intro x
  let a₀ : Space → ℝ := fun y => partial1 H₁ 1 y * partial1 H₂ 2 y - partial1 H₁ 2 y * partial1 H₂ 1 y
  let a₁ : Space → ℝ := fun y => partial1 H₁ 2 y * partial1 H₂ 0 y - partial1 H₁ 0 y * partial1 H₂ 2 y
  let a₂ : Space → ℝ := fun y => partial1 H₁ 0 y * partial1 H₂ 1 y - partial1 H₁ 1 y * partial1 H₂ 0 y
  have ha₀ : DifferentiableAt ℝ a₀ x :=
    ((partial1_differentiableAt H₁ h1 1 x).mul (partial1_differentiableAt H₂ h2 2 x)).sub
      ((partial1_differentiableAt H₁ h1 2 x).mul (partial1_differentiableAt H₂ h2 1 x))
  have ha₁ : DifferentiableAt ℝ a₁ x :=
    ((partial1_differentiableAt H₁ h1 2 x).mul (partial1_differentiableAt H₂ h2 0 x)).sub
      ((partial1_differentiableAt H₁ h1 0 x).mul (partial1_differentiableAt H₂ h2 2 x))
  have ha₂ : DifferentiableAt ℝ a₂ x :=
    ((partial1_differentiableAt H₁ h1 0 x).mul (partial1_differentiableAt H₂ h2 1 x)).sub
      ((partial1_differentiableAt H₁ h1 1 x).mul (partial1_differentiableAt H₂ h2 0 x))
  change TevSolenoidalBulk.divergence (fun y => a₀ y • e 0 + a₁ y • e 1 + a₂ y • e 2) x = 0
  rw [divergence_three_basis_sum a₀ a₁ a₂ x ha₀ ha₁ ha₂]
  dsimp [a₀, a₁, a₂]
  rw [fderiv_mul_sub_apply _ _ _ _ x (e 0)
        (partial1_differentiableAt H₁ h1 1 x) (partial1_differentiableAt H₁ h1 2 x)
        (partial1_differentiableAt H₂ h2 2 x) (partial1_differentiableAt H₂ h2 1 x)]
  rw [fderiv_mul_sub_apply _ _ _ _ x (e 1)
        (partial1_differentiableAt H₁ h1 2 x) (partial1_differentiableAt H₁ h1 0 x)
        (partial1_differentiableAt H₂ h2 0 x) (partial1_differentiableAt H₂ h2 2 x)]
  rw [fderiv_mul_sub_apply _ _ _ _ x (e 2)
        (partial1_differentiableAt H₁ h1 0 x) (partial1_differentiableAt H₁ h1 1 x)
        (partial1_differentiableAt H₂ h2 1 x) (partial1_differentiableAt H₂ h2 0 x)]
  change (partial1 (partial1 H₁ 1) 0 x * partial1 H₂ 2 x + partial1 H₁ 1 x * partial1 (partial1 H₂ 2) 0 x) -
         (partial1 (partial1 H₁ 2) 0 x * partial1 H₂ 1 x + partial1 H₁ 2 x * partial1 (partial1 H₂ 1) 0 x) +
         ((partial1 (partial1 H₁ 2) 1 x * partial1 H₂ 0 x + partial1 H₁ 2 x * partial1 (partial1 H₂ 0) 1 x) -
          (partial1 (partial1 H₁ 0) 1 x * partial1 H₂ 2 x + partial1 H₁ 0 x * partial1 (partial1 H₂ 2) 1 x)) +
         ((partial1 (partial1 H₁ 0) 2 x * partial1 H₂ 1 x + partial1 H₁ 0 x * partial1 (partial1 H₂ 1) 2 x) -
          (partial1 (partial1 H₁ 1) 2 x * partial1 H₂ 0 x + partial1 H₁ 1 x * partial1 (partial1 H₂ 0) 2 x)) = 0
  rw [schwarz_c2_comm H₁ h1 1 0 x, schwarz_c2_comm H₁ h1 2 0 x, schwarz_c2_comm H₁ h1 2 1 x]
  rw [schwarz_c2_comm H₂ h2 1 0 x, schwarz_c2_comm H₂ h2 2 0 x, schwarz_c2_comm H₂ h2 2 1 x]
  ring

-- ТЕОРЕМА 4 (ПОЛНОЕ СНЯТИЕ ПОСТУЛАТА МИЗЕСА ИЗ C²-ГЛАДКОСТИ НАМБУ):
-- Для любых C²-потенциалов H₁, H₂ индуцированная пластическая деформация строго изохорна,
-- а диссипация Мизеса σ : D тождественно не зависит от гидростатического давления.
theorem nambu_c2_isochoric_mises_bridge (H₁ H₂ : Space → ℝ)
    (h1 : ContDiff ℝ 2 H₁) (h2 : ContDiff ℝ 2 H₂) (x : Space)
    (σ : TevNambuMises.SymmetricStressTensor) (D : TevNambuMises.StrainRateTensor)
    (hdx : D.dxx = fderiv ℝ (nambuVectorField H₁ H₂) x (e 0) 0)
    (hdy : D.dyy = fderiv ℝ (nambuVectorField H₁ H₂) x (e 1) 1)
    (hdz : D.dzz = fderiv ℝ (nambuVectorField H₁ H₂) x (e 2) 2) :
    TevNambuMises.volumetric_strain_rate D = 0 ∧
    TevNambuMises.tensor_contract σ D =
      TevNambuMises.tensor_contract (TevNambuMises.deviator_tensor σ) D := by
  have h_div : TevSolenoidalBulk.divergence (nambuVectorField H₁ H₂) x = 0 :=
    nambu_c2_solenoidal H₁ H₂ h1 h2 x
  have h_iso : TevNambuMises.volumetric_strain_rate D = 0 := by
    dsimp [TevNambuMises.volumetric_strain_rate]
    rw [hdx, hdy, hdz]
    dsimp [TevSolenoidalBulk.divergence] at h_div
    rw [Fin.sum_univ_three] at h_div
    exact h_div
  exact ⟨h_iso, TevNambuMises.mises_dissipation_pressure_independent σ D h_iso⟩

-- ============================================================================
-- 3. Закрытие отсутствия магнитных монополей для реальных C²-полей смещения
-- ============================================================================

/-- Координатная компонента векторного поля u : Space → Space как скалярная функция -/
def compField (u : Space → Space) (i : Fin 3) (x : Space) : ℝ :=
  EuclideanSpace.proj i (u x)

lemma compField_c2 (u : Space → Space) (hu : ContDiff ℝ 2 u) (i : Fin 3) :
    ContDiff ℝ 2 (compField u i) := by
  change ContDiff ℝ 2 ((EuclideanSpace.proj i) ∘ u)
  exact (EuclideanSpace.proj i).contDiff.comp hu

/-- Ротор векторного поля B = ∇ × u для произвольного векторного поля u : Space → Space -/
def curlField (u : Space → Space) (x : Space) : Space :=
  (partial1 (compField u 2) 1 x - partial1 (compField u 1) 2 x) • e 0 +
  (partial1 (compField u 0) 2 x - partial1 (compField u 2) 0 x) • e 1 +
  (partial1 (compField u 1) 0 x - partial1 (compField u 0) 1 x) • e 2

-- ТЕОРЕМА 5 (ЗАПРЕТ МАГНИТНЫХ МОНОПОЛЕЙ ДЛЯ РЕАЛЬНЫХ C²-ПОЛЕЙ СМЕЩЕНИЯ):
-- Для любого дважды непрерывно дифференцируемого поля смещений u : Space → Space
-- поле B = ∇ × u строго соленоидально: ∇ · B ≡ 0 (без ручных допущений Шварца).
theorem curl_c2_solenoidal (u : Space → Space) (hu : ContDiff ℝ 2 u) :
    TevSolenoidalBulk.SolenoidalField (curlField u) := by
  intro x
  have hc0 := compField_c2 u hu 0
  have hc1 := compField_c2 u hu 1
  have hc2 := compField_c2 u hu 2
  let b₀ : Space → ℝ := fun y => partial1 (compField u 2) 1 y - partial1 (compField u 1) 2 y
  let b₁ : Space → ℝ := fun y => partial1 (compField u 0) 2 y - partial1 (compField u 2) 0 y
  let b₂ : Space → ℝ := fun y => partial1 (compField u 1) 0 y - partial1 (compField u 0) 1 y
  have hb₀ : DifferentiableAt ℝ b₀ x :=
    (partial1_differentiableAt _ hc2 1 x).sub (partial1_differentiableAt _ hc1 2 x)
  have hb₁ : DifferentiableAt ℝ b₁ x :=
    (partial1_differentiableAt _ hc0 2 x).sub (partial1_differentiableAt _ hc2 0 x)
  have hb₂ : DifferentiableAt ℝ b₂ x :=
    (partial1_differentiableAt _ hc1 0 x).sub (partial1_differentiableAt _ hc0 1 x)
  change TevSolenoidalBulk.divergence (fun y => b₀ y • e 0 + b₁ y • e 1 + b₂ y • e 2) x = 0
  rw [divergence_three_basis_sum b₀ b₁ b₂ x hb₀ hb₁ hb₂]
  have hd0 : fderiv ℝ b₀ x (e 0) =
      partial1 (partial1 (compField u 2) 1) 0 x - partial1 (partial1 (compField u 1) 2) 0 x := by
    have h_sub : HasFDerivAt
        (fun y => partial1 (compField u 2) 1 y - partial1 (compField u 1) 2 y)
        (fderiv ℝ (partial1 (compField u 2) 1) x -
         fderiv ℝ (partial1 (compField u 1) 2) x) x :=
      (partial1_differentiableAt _ hc2 1 x).hasFDerivAt.sub
        (partial1_differentiableAt _ hc1 2 x).hasFDerivAt
    dsimp [b₀]
    rw [h_sub.fderiv]
    rfl
  have hd1 : fderiv ℝ b₁ x (e 1) =
      partial1 (partial1 (compField u 0) 2) 1 x - partial1 (partial1 (compField u 2) 0) 1 x := by
    have h_sub : HasFDerivAt
        (fun y => partial1 (compField u 0) 2 y - partial1 (compField u 2) 0 y)
        (fderiv ℝ (partial1 (compField u 0) 2) x -
         fderiv ℝ (partial1 (compField u 2) 0) x) x :=
      (partial1_differentiableAt _ hc0 2 x).hasFDerivAt.sub
        (partial1_differentiableAt _ hc2 0 x).hasFDerivAt
    dsimp [b₁]
    rw [h_sub.fderiv]
    rfl
  have hd2 : fderiv ℝ b₂ x (e 2) =
      partial1 (partial1 (compField u 1) 0) 2 x - partial1 (partial1 (compField u 0) 1) 2 x := by
    have h_sub : HasFDerivAt
        (fun y => partial1 (compField u 1) 0 y - partial1 (compField u 0) 1 y)
        (fderiv ℝ (partial1 (compField u 1) 0) x -
         fderiv ℝ (partial1 (compField u 0) 1) x) x :=
      (partial1_differentiableAt _ hc1 0 x).hasFDerivAt.sub
        (partial1_differentiableAt _ hc0 1 x).hasFDerivAt
    dsimp [b₂]
    rw [h_sub.fderiv]
    rfl
  rw [hd0, hd1, hd2]
  rw [schwarz_c2_comm (compField u 0) hc0 2 1 x]
  rw [schwarz_c2_comm (compField u 1) hc1 2 0 x]
  rw [schwarz_c2_comm (compField u 2) hc2 1 0 x]
  ring

end

end TevNambuSchwarzC2
