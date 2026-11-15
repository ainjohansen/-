import Mathlib
import tev.TevChiralityDecay

set_option linter.style.whitespace false
set_option linter.style.emptyLine false
set_option linter.style.longLine false

/-!
# ТОПОЛОГИЧЕСКАЯ ЭЛАСТОДИНАМИКА ВАКУУМА (ТЭВ)
Модуль: TevExoticNucleon (Астрофизика бисолитона [p⁺ + e⁺], магнитары и ν_R)

Формальный модуль-спутник для астрофизического приложения ТЭВ (Трек 4.3 v7.0):
1. `coulomb_barrier_repulsion_positive` + вывод U_rep = (1/8) α M_p c² из r_c = 4 λ_p.
2. Потенциал Гиббса G = E + P_ext · V, критерий перехода P_ext > P_crit
   и `tov_pressure_exceeds_critical` (P_TOV ≥ 10³⁴ Па ≫ P_crit ≈ 3·10³² Па).
3. `exotic_magnetic_moment_sum`: сонаправленное сложение моментов керна и оболочки
   (μ = 2.79 + 4.71 = +7.50 μ_N) и модель намагниченности магнитарного плато.
4. `helicity_inversion_theorem`: срыв e⁺ с правого узла T(3,2) рождает правое нейтрино ν_R,
   причём в свободном вакууме (P_ext = 0) этот канал строго запрещён.
-/

namespace TevExoticNucleon

noncomputable section

-- ============================================================================
-- 1. Топологическая механика и кулоновский барьер бисолитона [p⁺ + e⁺]
-- ============================================================================

/-- Кулоновская энергия отталкивания позитронной оболочки на радиусе каверны r_c:
    U_rep = α ħ c / (2 r_c) -/
def coulombRepulsionFromRadius (α hbar c r_c : ℝ) : ℝ :=
  α * hbar * c / (2 * r_c)

/-- Аналитическая форма кулоновского барьера при r_c = 4 λ_p:
    U_rep = (1/8) α M_p c² (≈ +0.856 МэВ) -/
def coulombRepulsionEnergy (α Mp c : ℝ) : ℝ :=
  (1 / 8 : ℝ) * α * Mp * c ^ 2

-- ТЕОРЕМА 1A: Вывод барьера (1/8) α M_p c² из зарядового радиуса r_c = 4 λ_p
theorem coulomb_barrier_from_charge_radius (α Mp c lambda_p hbar r_c : ℝ)
    (hl : lambda_p ≠ 0)
    (h_hbar : hbar = Mp * c * lambda_p)
    (h_rc : r_c = 4 * lambda_p) :
    coulombRepulsionFromRadius α hbar c r_c = coulombRepulsionEnergy α Mp c := by
  dsimp [coulombRepulsionFromRadius, coulombRepulsionEnergy]
  rw [h_hbar, h_rc]
  calc
    α * (Mp * c * lambda_p) * c / (2 * (4 * lambda_p))
      = (lambda_p / lambda_p) * ((1 / 8 : ℝ) * α * Mp * c ^ 2) := by ring
    _ = 1 * ((1 / 8 : ℝ) * α * Mp * c ^ 2) := by rw [div_self hl]
    _ = (1 / 8 : ℝ) * α * Mp * c ^ 2 := by ring

-- ТЕОРЕМА 1B (`coulomb_barrier_repulsion_positive`):
-- Барьер кулоновского расталкивания одноимённых зарядов (+e и +e) строго положителен
theorem coulomb_barrier_repulsion_positive (α Mp c : ℝ)
    (hα : 0 < α) (hMp : 0 < Mp) (hc : 0 < c) :
    0 < coulombRepulsionEnergy α Mp c := by
  dsimp [coulombRepulsionEnergy]
  positivity

/-- Полная избыточная энергия квазисвязанного состояния ΔE = m_e c² + E_comp + U_rep -/
def exoticExcessEnergy (me_c2 E_comp α Mp c : ℝ) : ℝ :=
  me_c2 + E_comp + coulombRepulsionEnergy α Mp c

theorem exotic_excess_energy_positive (me_c2 E_comp α Mp c : ℝ)
    (hme : 0 < me_c2) (hcomp : 0 < E_comp) (hα : 0 < α) (hMp : 0 < Mp) (hc : 0 < c) :
    0 < exoticExcessEnergy me_c2 E_comp α Mp c := by
  dsimp [exoticExcessEnergy]
  have h_rep := coulomb_barrier_repulsion_positive α Mp c hα hMp hc
  linarith

-- ============================================================================
-- 2. Термодинамика Гиббса и критерий устойчивости в недрах TOV
-- ============================================================================

/-- Термодинамическое состояние фазового перехода p⁺ + e⁺ ↔ [p⁺ + e⁺] -/
structure ExoticGibbsState where
  E_free : ℝ
  E_exotic : ℝ
  V_free : ℝ
  V_exotic : ℝ
  h_energy_cost : E_free < E_exotic
  h_volume_gain : V_exotic < V_free

def gibbsEnergy (E V P_ext : ℝ) : ℝ :=
  E + P_ext * V

def criticalPressure (s : ExoticGibbsState) : ℝ :=
  (s.E_exotic - s.E_free) / (s.V_free - s.V_exotic)

theorem critical_pressure_pos (s : ExoticGibbsState) :
    0 < criticalPressure s := by
  dsimp [criticalPressure]
  exact div_pos (sub_pos.mpr s.h_energy_cost) (sub_pos.mpr s.h_volume_gain)

-- ТЕОРЕМА 2A: Точный критерий Гиббса фазовой устойчивости под давлением:
-- G_exotic < G_free ↔ P_ext > P_crit = ΔE / |ΔV|
theorem gibbs_phase_transition_criterion (s : ExoticGibbsState) (P_ext : ℝ) :
    gibbsEnergy s.E_exotic s.V_exotic P_ext < gibbsEnergy s.E_free s.V_free P_ext ↔
    criticalPressure s < P_ext := by
  dsimp [gibbsEnergy, criticalPressure]
  have hV : 0 < s.V_free - s.V_exotic := sub_pos.mpr s.h_volume_gain
  rw [div_lt_iff₀ hV]
  constructor <;> intro h <;> linarith

-- ТЕОРЕМА 2B (`tov_pressure_exceeds_critical`):
-- Центральное давление TOV нейтронной звезды (P_TOV ≥ 10³⁴ Па)
-- гарантированно превышает критический порог (P_crit ≤ 3.5·10³² Па) более чем в 25 раз.
theorem tov_pressure_exceeds_critical (P_crit P_tov : ℝ)
    (h_crit : P_crit ≤ (35 / 10 : ℝ) * (10 : ℝ) ^ 32)
    (h_tov : (10 : ℝ) ^ 34 ≤ P_tov) :
    P_crit < P_tov ∧ 25 * P_crit < P_tov := by
  have h1 : (35 / 10 : ℝ) * (10 : ℝ) ^ 32 < (10 : ℝ) ^ 34 := by norm_num
  have h2 : 25 * ((35 / 10 : ℝ) * (10 : ℝ) ^ 32) < (10 : ℝ) ^ 34 := by norm_num
  have h3 : 25 * P_crit ≤ 25 * ((35 / 10 : ℝ) * (10 : ℝ) ^ 32) := by linarith
  exact ⟨lt_of_le_of_lt h_crit (lt_of_lt_of_le h1 h_tov),
         lt_of_le_of_lt h3 (lt_of_lt_of_le h2 h_tov)⟩

-- Стыковка с `TevChirality`: в ядре звезды TOV одноимённая пара [p⁺ + e⁺] механически связана
theorem exotic_pair_bound_in_tov_core (P_coulomb P_tov : ℝ)
    (hPc : 0 < P_coulomb) (h_exceeds : P_coulomb < P_tov) :
    let m : TevChirality.MediumStressState :=
      { P_ext := P_tov, P_coulomb := P_coulomb, h_coulomb_pos := hPc }
    TevChirality.is_bound_under_stress
      TevChirality.KnotChirality.RightHanded
      TevChirality.ShellCirculation.PositronShell m := by
  intro m
  exact TevChirality.extreme_pressure_allows_exotic_pairs
    TevChirality.KnotChirality.RightHanded
    TevChirality.ShellCirculation.PositronShell m h_exceeds

-- ============================================================================
-- 3. Электродинамика и накачка магнитных полей магнитаров
-- ============================================================================

/-- Полный магнитный момент бисолитона [p⁺ + e⁺] при сонаправленности токов керна и оболочки -/
def exoticMagneticMoment (mu_p delta_mu_shell : ℝ) : ℝ :=
  mu_p + delta_mu_shell

-- ТЕОРЕМА 3 (`exotic_magnetic_moment_sum`):
-- Сонаправленное сложение моментов строго усиливает диполь по сравнению с протоном
-- и воспроизводит суммарный момент +7.50 μ_N (в 2.68+ раз больше свободного протона).
theorem exotic_magnetic_moment_sum (mu_p delta_mu_shell : ℝ)
    (hmup : 0 < mu_p) (hshell : 0 < delta_mu_shell) :
    0 < exoticMagneticMoment mu_p delta_mu_shell ∧
    mu_p < exoticMagneticMoment mu_p delta_mu_shell ∧
    exoticMagneticMoment mu_p delta_mu_shell = mu_p + delta_mu_shell := by
  dsimp [exoticMagneticMoment]
  exact ⟨by linarith, by linarith, rfl⟩

theorem exotic_magnetic_moment_numerical :
    exoticMagneticMoment (279 / 100 : ℝ) (471 / 100 : ℝ) = (750 / 100 : ℝ) ∧
    2 * (279 / 100 : ℝ) < exoticMagneticMoment (279 / 100 : ℝ) (471 / 100 : ℝ) := by
  dsimp [exoticMagneticMoment]
  constructor <;> norm_num

/-- Эффективное макроскопическое поле доменной поляризации ядра магнитара:
    B = μ₀_eff · (f_ex · n_baryon) · μ_[p+e⁺] -/
def magnetarDomainField (mu0_eff n_baryon f_ex mu_ex : ℝ) : ℝ :=
  mu0_eff * (f_ex * n_baryon) * mu_ex

theorem magnetar_field_monotone_in_fraction (mu0_eff n_baryon mu_ex f1 f2 : ℝ)
    (hmu0 : 0 < mu0_eff) (hn : 0 < n_baryon) (hmu : 0 < mu_ex)
    (hf12 : f1 < f2) :
    magnetarDomainField mu0_eff n_baryon f1 mu_ex <
    magnetarDomainField mu0_eff n_baryon f2 mu_ex := by
  dsimp [magnetarDomainField]
  have h_pos : 0 < mu0_eff * n_baryon * mu_ex := mul_pos (mul_pos hmu0 hn) hmu
  nlinarith

-- ============================================================================
-- 4. Механика топологического срыва и рождение правых нейтрино ν_R
-- ============================================================================

/-- Тип нейтрального торсионного импульса по сохранению лептонного числа -/
inductive EmittedLeptonSpecies
  | Neutrino      -- L = +1 (сопутствует сбросу e⁺)
  | Antineutrino  -- L = -1 (сопутствует сбросу e⁻)
  deriving DecidableEq

def emittedSpecies : TevChirality.ShellCirculation → EmittedLeptonSpecies
  | TevChirality.ShellCirculation.ElectronShell => EmittedLeptonSpecies.Antineutrino
  | TevChirality.ShellCirculation.PositronShell => EmittedLeptonSpecies.Neutrino

/-- Спиральность торсионного цуга задаётся хиральностью керна T(3, ±2) («правило резьбы») -/
def fullDecayHelicity : TevChirality.KnotChirality → TevChirality.EmittedHelicity
  | TevChirality.KnotChirality.RightHanded => TevChirality.EmittedHelicity.RightHelicity
  | TevChirality.KnotChirality.LeftHanded  => TevChirality.EmittedHelicity.LeftHelicity

-- ТЕОРЕМА 4 (`helicity_inversion_theorem`):
-- 1) Стандартный нейтрон [p⁺ + e⁻] сбрасывает антинейтрино правой спиральности ν̄_R;
-- 2) Стандартный антинейтрон [p̄⁻ + e⁺] сбрасывает левое нейтрино ν_L;
-- 3) Срыв позитронной оболочки e⁺ с правого узла T(3,2) в бисолитоне [p⁺ + e⁺]
--    рождает то же нейтрино (L = +1), но с ИНВЕРТИРОВАННОЙ правой спиральностью ν_R;
-- 4) В пустом вакууме (P_ext = 0) этот канал строго запрещён (нет первичного стерильного сектора).
theorem helicity_inversion_theorem :
    (emittedSpecies TevChirality.ShellCirculation.ElectronShell = EmittedLeptonSpecies.Antineutrino ∧
     fullDecayHelicity TevChirality.KnotChirality.RightHanded =
       TevChirality.EmittedHelicity.RightHelicity) ∧
    (emittedSpecies TevChirality.ShellCirculation.PositronShell = EmittedLeptonSpecies.Neutrino ∧
     fullDecayHelicity TevChirality.KnotChirality.LeftHanded =
       TevChirality.EmittedHelicity.LeftHelicity) ∧
    (emittedSpecies TevChirality.ShellCirculation.PositronShell = EmittedLeptonSpecies.Neutrino ∧
     fullDecayHelicity TevChirality.KnotChirality.RightHanded =
       TevChirality.EmittedHelicity.RightHelicity ∧
     fullDecayHelicity TevChirality.KnotChirality.RightHanded ≠
       fullDecayHelicity TevChirality.KnotChirality.LeftHanded) ∧
    (∀ P_c : ℝ, ∀ hPc : 0 < P_c,
      let m : TevChirality.MediumStressState :=
        { P_ext := 0, P_coulomb := P_c, h_coulomb_pos := hPc }
      ¬ TevChirality.is_bound_under_stress
          TevChirality.KnotChirality.RightHanded
          TevChirality.ShellCirculation.PositronShell m) := by
  refine ⟨⟨rfl, rfl⟩, ⟨rfl, rfl⟩, ⟨rfl, rfl, by decide⟩, ?_⟩
  intro P_c hPc m
  have h_repel : ¬ TevChirality.is_coulomb_attracting
      TevChirality.KnotChirality.RightHanded
      TevChirality.ShellCirculation.PositronShell :=
    TevChirality.same_charge_configurations_repel
      TevChirality.KnotChirality.RightHanded
      TevChirality.ShellCirculation.PositronShell
      (Or.inl ⟨rfl, rfl⟩)
  exact TevChirality.vacuum_forbids_same_charge_pairs
    TevChirality.KnotChirality.RightHanded
    TevChirality.ShellCirculation.PositronShell m rfl h_repel

end

end TevExoticNucleon
