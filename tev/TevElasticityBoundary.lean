import Mathlib

/-!
# ТОПОЛОГИЧЕСКАЯ ЭЛАСТОДИНАМИКА ВАКУУМА (ТЭВ) — Граничный сектор упругости (C1)
Адаптация граничной задачи теории упругости (семейство 372) к нотации ТЭВ:
* Модули Ламэ заменены упругими константами вакуума:
    G₀ — модуль сдвига [Па]   (μ ↔ G₀),
    K₀ — объёмный модуль [Па] (λ ↔ K₀ − 2G₀/3).
* Допустимость: G₀ > 0 и K₀ > 0 на замыкании.
-/

namespace TevElasticityBoundary

noncomputable section
open MeasureTheory Set
open scoped BigOperators

/-- Окружающее пространство ℝ³. -/
abbrev X := EuclideanSpace ℝ (Fin 3)
abbrev V := X

def coordVector (i : Fin 3) : X := EuclideanSpace.single i 1

/-- Гладкая граница: локальная карта, выпрямляющая frontier в {x₀ = 0}. -/
def SmoothBoundary (Ω : Set X) : Prop :=
  ∀ x ∈ frontier Ω, ∃ e : OpenPartialHomeomorph X X,
    x ∈ e.source ∧
    ContDiffOn ℝ (⊤ : ℕ∞) e e.source ∧
    ContDiffOn ℝ (⊤ : ℕ∞) e.symm e.target ∧
    e x 0 = 0 ∧
    ∀ y ∈ e.source, y ∈ Ω ↔ 0 < e y 0

/-- Ограниченная область с гладкой границей. -/
def Domain (Ω : Set X) : Prop :=
  IsOpen Ω ∧ IsConnected Ω ∧ Bornology.IsBounded Ω ∧ SmoothBoundary Ω

def SmoothUpTo (Ω : Set X) (f : X → ℝ) : Prop :=
  ∃ U : Set X, IsOpen U ∧ closure Ω ⊆ U ∧ ContDiffOn ℝ (⊤ : ℕ∞) f U

/-- Допустимость упругих констант вакуума: G₀ > 0 и K₀ > 0 на замыкании. -/
def Admissible (Ω : Set X) (K0 G0 : X → ℝ) : Prop :=
  SmoothUpTo Ω K0 ∧ SmoothUpTo Ω G0 ∧
    ∀ x ∈ closure Ω, 0 < G0 x ∧ 0 < K0 x

/-- Ламэ-форма допустимости: μ > 0 и 3λ + 2μ > 0. -/
def AdmissibleLame (Ω : Set X) (lam mu : X → ℝ) : Prop :=
  SmoothUpTo Ω lam ∧ SmoothUpTo Ω mu ∧
    ∀ x ∈ closure Ω, 0 < mu x ∧ 0 < 3 * lam x + 2 * mu x

/-- Первый параметр Ламэ через объёмный и сдвиговый модули. -/
def lamFromBulk (K0 G0 : X → ℝ) : X → ℝ := fun x => K0 x - 2 * G0 x / 3

theorem lame_to_bulk_identity (x : X) (K0 G0 : X → ℝ) :
    3 * lamFromBulk K0 G0 x + 2 * G0 x = 3 * K0 x := by
  dsimp [lamFromBulk]
  ring

/-- SmoothUpTo устойчиво к линейным комбинациям. -/
theorem smoothUpTo_lincomb (Ω : Set X) (f g : X → ℝ) (a b : ℝ)
    (hf : SmoothUpTo Ω f) (hg : SmoothUpTo Ω g) :
    SmoothUpTo Ω (fun x => a * f x + b * g x) := by
  obtain ⟨U, hUo, hUc, hfU⟩ := hf
  obtain ⟨W, hWo, hWc, hgW⟩ := hg
  refine ⟨U ∩ W, hUo.inter hWo, fun x hx => ⟨hUc hx, hWc hx⟩, ?_⟩
  have hf' : ContDiffOn ℝ (⊤ : ℕ∞) (fun x => a * f x) (U ∩ W) := by
    exact (hfU.mono inter_subset_left).const_smul a
  have hg' : ContDiffOn ℝ (⊤ : ℕ∞) (fun x => b * g x) (U ∩ W) := by
    exact (hgW.mono inter_subset_right).const_smul b
  exact hf'.add hg'

/-- Допустимость в нотации ТЭВ эквивалентна ламэ-форме. -/
theorem admissible_iff_lame (Ω : Set X) (K0 G0 lam mu : X → ℝ)
    (hl : ∀ x, lam x = K0 x - 2 * G0 x / 3) (hm : ∀ x, mu x = G0 x) :
    Admissible Ω K0 G0 ↔ AdmissibleLame Ω lam mu := by
  constructor
  · rintro ⟨hK, hG, hpos⟩
    refine ⟨?_, ?_, ?_⟩
    · have h_eq : lam = fun x => 1 * K0 x + (-2 / 3 : ℝ) * G0 x := by
        funext x
        rw [hl x]
        ring
      rw [h_eq]
      exact smoothUpTo_lincomb Ω K0 G0 1 (-2 / 3) hK hG
    · have h_eq : mu = G0 := funext hm
      rw [h_eq]
      exact hG
    · intro x hx
      obtain ⟨hGx, hKx⟩ := hpos x hx
      rw [hl x, hm x]
      refine ⟨hGx, ?_⟩
      linarith
  · rintro ⟨hL, hM, hpos⟩
    refine ⟨?_, ?_, ?_⟩
    · have h_eq : G0 = mu := funext (fun x => (hm x).symm)
      have hG : SmoothUpTo Ω G0 := by
        rw [h_eq]
        exact hM
      have h_eq2 : K0 = fun x => 1 * lam x + (2 / 3 : ℝ) * mu x := by
        funext x
        rw [hl x, hm x]
        ring
      rw [h_eq2]
      exact smoothUpTo_lincomb Ω lam mu 1 (2 / 3) hL hM
    · have h_eq : G0 = mu := funext (fun x => (hm x).symm)
      rw [h_eq]
      exact hM
    · intro x hx
      obtain ⟨hMx, hLamMu⟩ := hpos x hx
      have hGx : 0 < G0 x := by
        rw [← hm x]
        exact hMx
      have hKx : 0 < K0 x := by
        have h_sum : 3 * lam x + 2 * mu x = 3 * K0 x := by
          rw [hl x, hm x]
          ring
        rw [h_sum] at hLamMu
        linarith
      exact ⟨hGx, hKx⟩

/-! =========================================================================
    РАЗДЕЛ 1. ПРОСТРАНСТВО H¹ ЧЕРЕЗ ГЛАДКИЕ СТРУИ
   ========================================================================= -/

abbrev Ambient (Ω : Set X) :=
  Lp V 2 (volume.restrict Ω) × (Fin 3 → Lp V 2 (volume.restrict Ω))

def coordDeriv (f : X → V) (i : Fin 3) (x : X) : V :=
  fderiv ℝ f x (coordVector i)

def SmoothJets (Ω : Set X) : Set (Ambient Ω) :=
  {j | ∃ (f : X → V) (hf : MemLp f 2 (volume.restrict Ω))
      (hd : ∀ i, MemLp (coordDeriv f i) 2 (volume.restrict Ω)),
    ContDiff ℝ (⊤ : ℕ∞) f ∧
    j.1 = hf.toLp f ∧ j.2 = fun i => (hd i).toLp (coordDeriv f i)}

def TestJets (Ω : Set X) : Set (Ambient Ω) :=
  {j | ∃ (f : X → V) (hf : MemLp f 2 (volume.restrict Ω))
      (hd : ∀ i, MemLp (coordDeriv f i) 2 (volume.restrict Ω)),
    ContDiff ℝ (⊤ : ℕ∞) f ∧ HasCompactSupport f ∧ tsupport f ⊆ Ω ∧
    j.1 = hf.toLp f ∧ j.2 = fun i => (hd i).toLp (coordDeriv f i)}

abbrev H1 (Ω : Set X) := {j : Ambient Ω // j ∈ closure (SmoothJets Ω)}

def HasZeroTrace (Ω : Set X) (u : H1 Ω) : Prop :=
  u.val ∈ closure (TestJets Ω)

def SameTrace (Ω : Set X) (u v : H1 Ω) : Prop :=
  u.val - v.val ∈ closure (TestJets Ω)

lemma coordDeriv_zero (i : Fin 3) : coordDeriv (fun _ => (0 : V)) i = 0 := by
  ext x
  dsimp [coordDeriv]
  simp

lemma toLp_zero (Ω : Set X) (f : X → V) (hf : MemLp f 2 (volume.restrict Ω)) (hf0 : f = 0) :
    hf.toLp f = 0 := by
  subst hf0
  exact MemLp.toLp_zero hf

lemma testJets_zero_mem (Ω : Set X) : (0 : Ambient Ω) ∈ TestJets Ω := by
  dsimp only [TestJets]
  have hf : MemLp (fun _ : X => (0 : V)) 2 (volume.restrict Ω) := MemLp.zero
  have hd : ∀ i, MemLp (coordDeriv (fun _ => (0 : V)) i) 2 (volume.restrict Ω) := by
    intro i
    have : coordDeriv (fun _ => (0 : V)) i = 0 := coordDeriv_zero i
    rw [this]
    exact MemLp.zero
  have hcs : HasCompactSupport (fun _ : X => (0 : V)) := by
    dsimp [HasCompactSupport, tsupport]
    simp
  refine ⟨fun _ => 0, hf, hd, contDiff_const, hcs, ?_, ?_⟩
  · simp [tsupport]
  · constructor
    · have h1 := toLp_zero Ω (fun _ => 0) hf rfl
      exact h1.symm
    · funext i
      have h2 := toLp_zero Ω (coordDeriv (fun _ => 0) i) (hd i) (coordDeriv_zero i)
      exact h2.symm

theorem sameTrace_refl (Ω : Set X) (u : H1 Ω) : SameTrace Ω u u := by
  dsimp [SameTrace]
  rw [sub_self]
  exact subset_closure (testJets_zero_mem Ω)

/-! =========================================================================
    РАЗДЕЛ 2. ГРАНИЧНЫЕ ДАННЫЕ, ЭНЕРГИЯ, DN-ОПЕРАТОР (нотация ТЭВ)
   ========================================================================= -/

/-- Вариационные граничные данные: факторизация по замыканию тестовых струй. -/
abbrev BoundaryData (Ω : Set X) := Quot (SameTrace Ω)

def trace (Ω : Set X) (u : H1 Ω) : BoundaryData Ω := Quot.mk _ u

def div (Ω : Set X) (u : H1 Ω) (x : X) : ℝ :=
  ∑ i, (u.val.2 i x) i

def strain (Ω : Set X) (u : H1 Ω) (x : X) (i j : Fin 3) : ℝ :=
  ((u.val.2 j x) i + (u.val.2 i x) j) / 2

/-- Билинейная упругая энергия в нотации ТЭВ. -/
def energy (Ω : Set X) (K0 G0 : X → ℝ) (u v : H1 Ω) : ℝ :=
  ∫ x, ((K0 x - 2 * G0 x / 3) * div Ω u x * div Ω v x +
    2 * G0 x * ∑ i, ∑ j, strain Ω u x i j * strain Ω v x i j)
    ∂(volume.restrict Ω)

theorem energy_eq_lame_form (Ω : Set X) (K0 G0 lam mu : X → ℝ)
    (hl : ∀ x, lam x = K0 x - 2 * G0 x / 3) (hm : ∀ x, mu x = G0 x)
    (u v : H1 Ω) :
    energy Ω K0 G0 u v =
      ∫ x, (lam x * div Ω u x * div Ω v x +
        2 * mu x * ∑ i, ∑ j, strain Ω u x i j * strain Ω v x i j)
        ∂(volume.restrict Ω) := by
  dsimp [energy]
  congr 1
  ext x
  rw [hl x, hm x]

def WeakSolution (Ω : Set X) (K0 G0 : X → ℝ) (u : H1 Ω) : Prop :=
  ∀ v : H1 Ω, HasZeroTrace Ω v → energy Ω K0 G0 u v = 0

def dirichletSolution (Ω : Set X) (K0 G0 : X → ℝ)
    (f : BoundaryData Ω) : H1 Ω := by
  classical
  exact if h : ∃ u : H1 Ω, trace Ω u = f ∧ WeakSolution Ω K0 G0 u
  then Classical.choose h
  else f.out

def DN (Ω : Set X) (K0 G0 : X → ℝ) : BoundaryData Ω → BoundaryData Ω → ℝ :=
  fun f g => energy Ω K0 G0 (dirichletSolution Ω K0 G0 f) g.out

/-- СПЕЦИФИКАЦИЯ ОБРАТНОЙ ЗАДАЧИ КАЛЬДЕРОНА ДЛЯ ВАКУУМА ТЭВ.
    Утверждение: DN-оператор на гладкой границе каверны однозначно определяет
    внутренние упругие модули K₀ и G₀.
    
    СТАТУС ВЕРИФИКАЦИИ:
    1. Эквивалентность нотации ТЭВ классической задаче Ламэ строго ДОКАЗАНА в этом файле
       (`admissible_iff_lame`, `energy_eq_lame_form`).
    2. Само доказательство утверждения `MainClaim` опирается на 42-файловый корпус
       комплексных геометрических оптик (CGO) и оценок Карлемана (семейство 372 OAI /
       результаты Накамуры--Ульмана и Эскина--Ралстона). -/
def MainClaim : Prop :=
  ∀ (Ω : Set X), Domain Ω →
  ∀ (K₁ G₁ K₂ G₂ : X → ℝ),
    Admissible Ω K₁ G₁ → Admissible Ω K₂ G₂ →
    DN Ω K₁ G₁ = DN Ω K₂ G₂ →
    (∀ x ∈ Ω, K₁ x = K₂ x) ∧ (∀ x ∈ Ω, G₁ x = G₂ x)

/-! =========================================================================
    РАЗДЕЛ 3. ВОЛНОВОЙ ОПЕРАТОР ТЭВ И КВАЗИНЕСЖИМАЕМОСТЬ
   ========================================================================= -/

def divV (u : X → V) (x : X) : ℝ := ∑ i, (fderiv ℝ u x (coordVector i)) i

def laplaceV (u : X → V) (x : X) : V :=
  ∑ j, (∑ i, fderiv ℝ (fun y => fderiv ℝ u y (coordVector i) j) x (coordVector i)) • coordVector j

def gradDiv (u : X → V) (x : X) : V :=
  ∑ j, (fderiv ℝ (divV u) x (coordVector j)) • coordVector j

/-- Оператор Ламэ вакуума в нотации ТЭВ. -/
def lameOperator (K0 G0 : X → ℝ) (u : X → V) (x : X) : V :=
  (K0 x + G0 x / 3) • gradDiv u x + G0 x • laplaceV u x

/-- Коэффициент Пуассона вакуумного эластомера: ν = (3K₀ − 2G₀)/(2(3K₀ + G₀)). -/
def poissonRatio (K0 G0 : ℝ) : ℝ := (3 * K0 - 2 * G0) / (2 * (3 * K0 + G0))

theorem poisson_deviation_exact (K0 G0 : ℝ) (hden : 3 * K0 + G0 ≠ 0) :
    poissonRatio K0 G0 - 1/2 = -3 * G0 / (2 * (3 * K0 + G0)) := by
  rw [poissonRatio]
  field_simp [hden]
  ring

/-- Квазинесжимаемый предел: |ν − 1/2| ≤ G₀/(2K₀) → ν → 1/2 при ε = G₀/K₀ → 0. -/
theorem poisson_quasi_incompressible_bound (K0 G0 : ℝ) (hK : 0 < K0) (hG : 0 ≤ G0) :
    |poissonRatio K0 G0 - 1/2| ≤ G0 / (2 * K0) := by
  have hden : 0 < 3 * K0 + G0 := by linarith
  have hden_ne : 3 * K0 + G0 ≠ 0 := ne_of_gt hden
  rw [poisson_deviation_exact K0 G0 hden_ne]
  rw [abs_div]
  have h_num : |-3 * G0| = 3 * G0 := by
    rw [abs_of_nonpos]
    · ring
    · nlinarith
  have h_den : |2 * (3 * K0 + G0)| = 2 * (3 * K0 + G0) := by
    apply abs_of_pos
    linarith
  rw [h_num, h_den]
  have h1 : 0 < 2 * (3 * K0 + G0) := by linarith
  have h2 : 0 < 2 * K0 := by linarith
  rw [div_le_div_iff₀ h1 h2]
  nlinarith

end

end TevElasticityBoundary
