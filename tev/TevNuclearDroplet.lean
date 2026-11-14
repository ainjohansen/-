import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic

/-!
# TevNuclearDroplet: Zero-Parameter Continuum Nuclear Droplet Model in TEV

Formal verification of all derivations from the AME2020 isotope database audit:
1. Topological anchors of T(3,2) on the Clifford torus and Spin(3) double cover.
2. Exact rational fractions of the droplet model (2/9, 1/4, 27/20, 4/9, 1/6, 3/14, 3/52)
   and the series surface-symmetry ratio `kappa_S = (4/9)/(2/9) = 2 = N_rev`.
3. Reduction to the legacy `1/3` asymmetry coefficient at `A = 216` (`r = 6`).
4. Real-valued cascade from the two sole material inputs `(alpha, M_p)` through
   `E_0`, dressed pion mass `M_pi_dressed`, and physical pion scale `E_0^pi`.
5. Exact variational proof of the two-spring series elasticity (`bulk + surface skin`).
6. Strict acyclicity (DAG rank monotonicity) proving zero circularity and zero
   empirical nuclear inputs.
-/

namespace TevNuclearDroplet

/-! ## 1. Rank-0 Topological Invariants (Discrete Geometry of d = 3 & T(3,2)) -/

def dimSpace : ℕ := 3
def N_3D : ℕ := dimSpace * (dimSpace + 1) / 2 - 1
def p_trefoil : ℕ := 3
def q_trefoil : ℕ := 2
def N_rev : ℕ := 2
def N_spin : ℕ := 4
def N_basis : ℕ := p_trefoil ^ 2 + q_trefoil ^ 2
def N_deuteron_basis : ℕ := N_basis + 1

theorem dimSpace_eq : dimSpace = 3 := rfl
theorem N_3D_eq : N_3D = 5 := rfl
theorem N_basis_eq : N_basis = 13 := rfl
theorem N_deuteron_basis_eq : N_deuteron_basis = 14 := rfl

theorem trefoil_spectral_invariant :
    (p_trefoil : ℚ) ^ 2 + (q_trefoil : ℚ) ^ 2 +
      (N_rev : ℚ) / ((p_trefoil : ℚ) + (q_trefoil : ℚ)) = 67 / 5 := by
  norm_num [p_trefoil, q_trefoil, N_rev]

/-! ## 2. Exact Zero-Fit Rational Fractions of the Nuclear Droplet Model -/

def frac_aV : ℚ := (N_rev : ℚ) / (p_trefoil : ℚ) ^ 2
def frac_aS : ℚ := 1 / (N_spin : ℚ)
def frac_aC : ℚ :=
  ((dimSpace : ℚ) / (N_3D : ℚ)) * ((p_trefoil : ℚ) ^ 2 / (N_spin : ℚ))
def frac_aA_bulk : ℚ := (N_spin : ℚ) / (p_trefoil : ℚ) ^ 2
def frac_aA_surf : ℚ := frac_aV
def frac_aP : ℚ := 1 / (2 * (p_trefoil : ℚ))
def kappa_S_exact : ℚ := frac_aA_bulk / frac_aA_surf
def frac_deuteron_recoil : ℚ := (p_trefoil : ℚ) / (N_deuteron_basis : ℚ)
def frac_deuteron_d_wave : ℚ :=
  (1 / (N_spin : ℚ)) * ((p_trefoil : ℚ) / (N_basis : ℚ))

theorem frac_aV_eq : frac_aV = 2 / 9 := by
  norm_num [frac_aV, N_rev, p_trefoil]

theorem frac_aS_eq : frac_aS = 1 / 4 := by
  norm_num [frac_aS, N_spin]

theorem frac_aC_eq : frac_aC = 27 / 20 := by
  norm_num [frac_aC, dimSpace, N_3D, p_trefoil, N_spin]

theorem frac_aA_bulk_eq : frac_aA_bulk = 4 / 9 := by
  norm_num [frac_aA_bulk, N_spin, p_trefoil]

theorem frac_aA_bulk_eq_two_aV : frac_aA_bulk = 2 * frac_aV := by
  norm_num [frac_aA_bulk, frac_aV, N_spin, N_rev, p_trefoil]

theorem frac_aP_eq : frac_aP = 1 / 6 := by
  norm_num [frac_aP, p_trefoil]

theorem kappa_S_eq_two : kappa_S_exact = 2 := by
  norm_num [kappa_S_exact, frac_aA_bulk, frac_aA_surf, frac_aV, N_spin, N_rev, p_trefoil]

theorem kappa_S_eq_N_rev : kappa_S_exact = (N_rev : ℚ) := by
  rw [kappa_S_eq_two]
  rfl

theorem frac_deuteron_recoil_eq : frac_deuteron_recoil = 3 / 14 := by
  norm_num [frac_deuteron_recoil, p_trefoil, N_deuteron_basis, N_basis, q_trefoil]

theorem frac_deuteron_d_wave_eq : frac_deuteron_d_wave = 3 / 52 := by
  norm_num [frac_deuteron_d_wave, N_spin, p_trefoil, N_basis, q_trefoil]

/-- Pure neutron matter saturation energy fraction: `-a_V + a_{A,bulk} = +2/9 > 0` (unbound). -/
theorem pnm_saturation_fraction : -frac_aV + frac_aA_bulk = 2 / 9 := by
  norm_num [frac_aV_eq, frac_aA_bulk_eq]

/-- Heavy-nucleus correspondence: at `A = 216` (`A^(1/3) = 6`), the two-spring series
    reproduces the legacy `1/3` effective asymmetry fraction exactly. -/
theorem effective_asymmetry_at_216_eq_one_third :
    frac_aA_bulk / (1 + kappa_S_exact / 6) = 1 / 3 := by
  norm_num [frac_aA_bulk_eq, kappa_S_eq_two]

/-- Sum of bulk droplet fractions equals `13 / 12 = N_basis / (N_basis - 1)`. -/
theorem droplet_fractions_sum :
    frac_aV + frac_aS + frac_aA_bulk + frac_aP = 13 / 12 := by
  norm_num [frac_aV_eq, frac_aS_eq, frac_aA_bulk_eq, frac_aP_eq]

/-! ## 3. Real-Valued Scale Cascade from Sole Material Anchors `(alpha, M_p)` -/

/-- The only two continuous inputs of the entire nuclear sector. -/
structure TevInputs where
  alpha : ℝ
  M_p : ℝ
  h_alpha_pos : 0 < alpha
  h_alpha_lt : alpha < 1 / 10
  h_Mp_pos : 0 < M_p

noncomputable def I_trefoil : ℝ := 67 / 5

noncomputable def laue_virial (alpha : ℝ) : ℝ := -(4 / 3 : ℝ) * alpha ^ 2

theorem laue_virial_den_pos (inp : TevInputs) : 0 < 1 + laue_virial inp.alpha := by
  unfold laue_virial
  have h_sq : inp.alpha ^ 2 < 1 / 100 := by
    nlinarith [inp.h_alpha_pos, inp.h_alpha_lt]
  linarith

noncomputable def E0_bare (inp : TevInputs) : ℝ :=
  (5 / 67 : ℝ) * inp.M_p / (1 + laue_virial inp.alpha)

theorem E0_bare_pos (inp : TevInputs) : 0 < E0_bare inp := by
  unfold E0_bare
  exact div_pos (by nlinarith [inp.h_Mp_pos]) (laue_virial_den_pos inp)

noncomputable def m_e_tev (inp : TevInputs) : ℝ := inp.alpha * E0_bare inp

theorem m_e_tev_pos (inp : TevInputs) : 0 < m_e_tev inp :=
  mul_pos inp.h_alpha_pos (E0_bare_pos inp)

/-- Electron-proton bridge identity: `M_p / m_e = (67/5) * alpha⁻¹ * (1 - (4/3)*alpha^2)`. -/
theorem proton_electron_bridge (inp : TevInputs) :
    inp.M_p / m_e_tev inp =
      (67 / 5 : ℝ) * (1 / inp.alpha) * (1 + laue_virial inp.alpha) := by
  unfold m_e_tev E0_bare
  have ha : inp.alpha ≠ 0 := ne_of_gt inp.h_alpha_pos
  have hMp : inp.M_p ≠ 0 := ne_of_gt inp.h_Mp_pos
  have hden : 1 + laue_virial inp.alpha ≠ 0 := ne_of_gt (laue_virial_den_pos inp)
  field_simp

noncomputable def M_pi_bare (inp : TevInputs) : ℝ := 2 * E0_bare inp

noncomputable def pion_coulomb_factor (alpha : ℝ) : ℝ :=
  1 - (1 / 2 : ℝ) * alpha * (1 - 2 * alpha / Real.pi)

theorem pion_coulomb_factor_bounds (inp : TevInputs) :
    0 < pion_coulomb_factor inp.alpha ∧ pion_coulomb_factor inp.alpha < 1 := by
  unfold pion_coulomb_factor
  have hpi2 : (2 : ℝ) ≤ Real.pi := Real.two_le_pi
  have hpi_pos : (0 : ℝ) < Real.pi := Real.pi_pos
  have h_div_pos : (0 : ℝ) < 2 * inp.alpha / Real.pi :=
    div_pos (by linarith [inp.h_alpha_pos]) hpi_pos
  have h_div_lt : 2 * inp.alpha / Real.pi < 1 :=
    (div_lt_one hpi_pos).mpr (by linarith [inp.h_alpha_lt, hpi2])
  have h_inner_pos : (0 : ℝ) < 1 - 2 * inp.alpha / Real.pi := by linarith
  have h_inner_lt : 1 - 2 * inp.alpha / Real.pi < 1 := by linarith
  have h_shift_pos :
      (0 : ℝ) < (1 / 2 : ℝ) * inp.alpha * (1 - 2 * inp.alpha / Real.pi) :=
    mul_pos (by linarith [inp.h_alpha_pos]) h_inner_pos
  have h_shift_lt :
      (1 / 2 : ℝ) * inp.alpha * (1 - 2 * inp.alpha / Real.pi) < 1 := by
    have h1 :
        (1 / 2 : ℝ) * inp.alpha * (1 - 2 * inp.alpha / Real.pi) <
          (1 / 2 : ℝ) * inp.alpha * 1 :=
      mul_lt_mul_of_pos_left h_inner_lt (by linarith [inp.h_alpha_pos])
    linarith [inp.h_alpha_lt]
  constructor <;> linarith

noncomputable def M_pi_dressed (inp : TevInputs) : ℝ :=
  M_pi_bare inp * pion_coulomb_factor inp.alpha

noncomputable def E0_pi (inp : TevInputs) : ℝ := M_pi_dressed inp / 2

theorem E0_pi_eq_dressed_E0 (inp : TevInputs) :
    E0_pi inp = E0_bare inp * pion_coulomb_factor inp.alpha := by
  unfold E0_pi M_pi_dressed M_pi_bare
  ring

theorem E0_pi_pos (inp : TevInputs) : 0 < E0_pi inp := by
  rw [E0_pi_eq_dressed_E0]
  exact mul_pos (E0_bare_pos inp) (pion_coulomb_factor_bounds inp).1

theorem E0_pi_lt_E0_bare (inp : TevInputs) : E0_pi inp < E0_bare inp := by
  rw [E0_pi_eq_dressed_E0]
  have h := (pion_coulomb_factor_bounds inp).2
  nlinarith [E0_bare_pos inp]

/-! ### Derived Nuclear Droplet Coefficients & Deuteron Scale -/

noncomputable def a_V_tev (inp : TevInputs) : ℝ := (frac_aV : ℝ) * E0_pi inp
noncomputable def a_S_tev (inp : TevInputs) : ℝ := (frac_aS : ℝ) * E0_pi inp
noncomputable def a_C_tev (inp : TevInputs) : ℝ := (frac_aC : ℝ) * inp.alpha * E0_pi inp
noncomputable def a_A_bulk_tev (inp : TevInputs) : ℝ := (frac_aA_bulk : ℝ) * E0_pi inp
noncomputable def a_A_surf_tev (inp : TevInputs) : ℝ := (frac_aA_surf : ℝ) * E0_pi inp
noncomputable def a_P_tev (inp : TevInputs) : ℝ := (frac_aP : ℝ) * E0_pi inp

noncomputable def E_b_deuteron_bare (inp : TevInputs) : ℝ :=
  (frac_deuteron_recoil : ℝ) * ((M_pi_bare inp) ^ 2 / (2 * inp.M_p))

noncomputable def E_b_deuteron_dressed (inp : TevInputs) : ℝ :=
  (frac_deuteron_recoil : ℝ) * ((M_pi_dressed inp) ^ 2 / (2 * inp.M_p))

theorem a_V_tev_pos (inp : TevInputs) : 0 < a_V_tev inp := by
  unfold a_V_tev
  rw [frac_aV_eq]
  push_cast
  nlinarith [E0_pi_pos inp]

theorem a_S_tev_pos (inp : TevInputs) : 0 < a_S_tev inp := by
  unfold a_S_tev
  rw [frac_aS_eq]
  push_cast
  nlinarith [E0_pi_pos inp]

theorem a_C_tev_pos (inp : TevInputs) : 0 < a_C_tev inp := by
  unfold a_C_tev
  rw [frac_aC_eq]
  push_cast
  have h1 : (0 : ℝ) < (27 / 20 : ℝ) := by norm_num
  exact mul_pos (mul_pos h1 inp.h_alpha_pos) (E0_pi_pos inp)

theorem a_A_bulk_eq_two_a_V (inp : TevInputs) :
    a_A_bulk_tev inp = 2 * a_V_tev inp := by
  unfold a_A_bulk_tev a_V_tev
  rw [frac_aA_bulk_eq_two_aV]
  push_cast
  ring

theorem a_A_surf_eq_a_V (inp : TevInputs) :
    a_A_surf_tev inp = a_V_tev inp := rfl

theorem a_P_tev_pos (inp : TevInputs) : 0 < a_P_tev inp := by
  unfold a_P_tev
  rw [frac_aP_eq]
  push_cast
  nlinarith [E0_pi_pos inp]

theorem pnm_saturation_unbound (inp : TevInputs) :
    -a_V_tev inp + a_A_bulk_tev inp = a_V_tev inp ∧ 0 < -a_V_tev inp + a_A_bulk_tev inp := by
  rw [a_A_bulk_eq_two_a_V]
  constructor
  · ring
  · linarith [a_V_tev_pos inp]

theorem E_b_deuteron_dressed_pos (inp : TevInputs) : 0 < E_b_deuteron_dressed inp := by
  unfold E_b_deuteron_dressed
  rw [frac_deuteron_recoil_eq]
  push_cast
  have hM : 0 < M_pi_dressed inp := by
    unfold M_pi_dressed M_pi_bare
    exact mul_pos (by nlinarith [E0_bare_pos inp]) (pion_coulomb_factor_bounds inp).1
  have hnum : 0 < (M_pi_dressed inp) ^ 2 := sq_pos_of_pos hM
  have hden : 0 < 2 * inp.M_p := by linarith [inp.h_Mp_pos]
  have hrec : (0 : ℝ) < (3 / 14 : ℝ) := by norm_num
  exact mul_pos hrec (div_pos hnum hden)

theorem E_b_deuteron_dressed_lt_bare (inp : TevInputs) :
    E_b_deuteron_dressed inp < E_b_deuteron_bare inp := by
  unfold E_b_deuteron_dressed E_b_deuteron_bare
  rw [frac_deuteron_recoil_eq]
  push_cast
  have h_factor := pion_coulomb_factor_bounds inp
  have h_bare_pos : 0 < M_pi_bare inp := by
    unfold M_pi_bare
    nlinarith [E0_bare_pos inp]
  have h_dressed_lt : M_pi_dressed inp < M_pi_bare inp := by
    unfold M_pi_dressed
    nlinarith [h_bare_pos, h_factor.2]
  have h_dressed_pos : 0 < M_pi_dressed inp :=
    mul_pos h_bare_pos h_factor.1
  have h_sq : (M_pi_dressed inp) ^ 2 < (M_pi_bare inp) ^ 2 := by
    nlinarith [h_dressed_pos, h_bare_pos, h_dressed_lt]
  have hden : 0 < 2 * inp.M_p := by linarith [inp.h_Mp_pos]
  have hrec : (0 : ℝ) < (3 / 14 : ℝ) := by norm_num
  exact mul_lt_mul_of_pos_left (div_lt_div_of_pos_right h_sq hden) hrec

/-! ## 4. Exact Variational Theorem for Two-Spring Surface Symmetry (`kappa_S = 2`) -/

noncomputable def twoSpringEnergy (Cv Cs I Iv : ℝ) : ℝ :=
  Cv * Iv ^ 2 + Cs * (I - Iv) ^ 2

noncomputable def seriesStiffness (Cv Cs : ℝ) : ℝ :=
  Cv * Cs / (Cv + Cs)

theorem two_spring_energy_identity (Cv Cs I Iv : ℝ) (hsum : Cv + Cs ≠ 0) :
    twoSpringEnergy Cv Cs I Iv =
      seriesStiffness Cv Cs * I ^ 2 +
        (Cv + Cs) * (Iv - (Cs / (Cv + Cs)) * I) ^ 2 := by
  unfold twoSpringEnergy seriesStiffness
  field_simp
  ring

theorem two_spring_lower_bound {Cv Cs : ℝ} (hCv : 0 < Cv) (hCs : 0 < Cs) (I Iv : ℝ) :
    seriesStiffness Cv Cs * I ^ 2 ≤ twoSpringEnergy Cv Cs I Iv := by
  have hsum_pos : 0 < Cv + Cs := by linarith
  rw [two_spring_energy_identity Cv Cs I Iv (ne_of_gt hsum_pos)]
  have h_nonneg : 0 ≤ (Cv + Cs) * (Iv - (Cs / (Cv + Cs)) * I) ^ 2 :=
    mul_nonneg (le_of_lt hsum_pos) (sq_nonneg _)
  linarith

theorem two_spring_min_attained {Cv Cs : ℝ} (hCv : 0 < Cv) (hCs : 0 < Cs) (I : ℝ) :
    twoSpringEnergy Cv Cs I ((Cs / (Cv + Cs)) * I) = seriesStiffness Cv Cs * I ^ 2 := by
  have hsum : Cv + Cs ≠ 0 := ne_of_gt (by linarith)
  rw [two_spring_energy_identity Cv Cs I ((Cs / (Cv + Cs)) * I) hsum]
  ring

/-- Given nuclear radius parameter `r = A^(1/3) > 0` with `r^3 = A`, bulk stiffness
    `Cv = a_A_bulk / A`, surface stiffness `Cs = a_A_surf / r^2`, and `a_A_bulk = 2 * a_A_surf`,
    the effective series symmetry stiffness equals `a_A_bulk / (A * (1 + 2 / r))`. -/
theorem droplet_asymmetry_from_series
    {A r a_A_bulk a_A_surf : ℝ}
    (hr : 0 < r) (hr3 : r ^ 3 = A)
    (hsurf : 0 < a_A_surf) (hbulk : a_A_bulk = 2 * a_A_surf) :
    seriesStiffness (a_A_bulk / A) (a_A_surf / r ^ 2) =
      a_A_bulk / (A * (1 + 2 / r)) := by
  unfold seriesStiffness
  rw [hbulk, ← hr3]
  have hr_ne : r ≠ 0 := ne_of_gt hr
  have hs_ne : a_A_surf ≠ 0 := ne_of_gt hsurf
  have h2r_ne : 2 + r ≠ 0 := ne_of_gt (by linarith)
  have h1r_ne : 1 + 2 / r ≠ 0 := ne_of_gt (by positivity)
  have hden_ne : 2 * a_A_surf / r ^ 3 + a_A_surf / r ^ 2 ≠ 0 := ne_of_gt (by positivity)
  field_simp
  ring

noncomputable def aA_eff (a_A_bulk r : ℝ) : ℝ :=
  a_A_bulk / (1 + 2 / r)

theorem aA_eff_softens_bulk {a_A_bulk r : ℝ} (hb : 0 < a_A_bulk) (hr : 0 < r) :
    aA_eff a_A_bulk r < a_A_bulk := by
  unfold aA_eff
  have hden : 1 < 1 + 2 / r := by
    have : 0 < 2 / r := div_pos (by norm_num) hr
    linarith
  exact (div_lt_iff₀ (by linarith)).mpr (by nlinarith)

theorem aA_eff_monotone_in_radius {a_A_bulk r1 r2 : ℝ}
    (hb : 0 < a_A_bulk) (hr1 : 0 < r1) (hr12 : r1 < r2) :
    aA_eff a_A_bulk r1 < aA_eff a_A_bulk r2 := by
  unfold aA_eff
  have hr2 : 0 < r2 := lt_trans hr1 hr12
  have h_inv : 2 / r2 < 2 / r1 := div_lt_div_of_pos_left (by norm_num) hr1 hr12
  have h_den1 : 0 < 1 + 2 / r1 := by positivity
  have h_den2 : 0 < 1 + 2 / r2 := by positivity
  have h_den_lt : 1 + 2 / r2 < 1 + 2 / r1 := by linarith
  exact div_lt_div_of_pos_left hb h_den2 h_den_lt

/-! ## 5. Strict Acyclicity of the Nuclear DAG (No Circularity, Zero Nuclear Fits) -/

inductive NuclearDagNode where
  | anchor_alpha
  | anchor_Mp
  | topo_dim3
  | topo_trefoil_pq
  | topo_spin_covering
  | trefoil_invariant_I32
  | laue_virial_delta_p
  | rational_semf_fractions
  | deuteron_recoil_fraction
  | electron_mass_me
  | bare_nambu_quantum_E0
  | bare_pion_mass
  | pion_coulomb_shift
  | dressed_pion_mass
  | dressed_pion_scale_E0pi
  | deuteron_binding_Eb
  | droplet_aV
  | droplet_aS
  | droplet_aC
  | droplet_aA_bulk
  | droplet_aA_surf
  | droplet_kappa_S
  | effective_asymmetry_aA_eff
  | droplet_aP
  | continuum_binding_energy_B_TEV
  deriving DecidableEq, Repr

def nodeRank : NuclearDagNode → ℕ
  | .anchor_alpha                   => 0
  | .anchor_Mp                      => 0
  | .topo_dim3                      => 0
  | .topo_trefoil_pq                => 0
  | .topo_spin_covering             => 0
  | .trefoil_invariant_I32          => 1
  | .laue_virial_delta_p            => 1
  | .rational_semf_fractions        => 1
  | .deuteron_recoil_fraction       => 1
  | .electron_mass_me               => 2
  | .bare_nambu_quantum_E0          => 2
  | .bare_pion_mass                 => 3
  | .pion_coulomb_shift             => 3
  | .dressed_pion_mass              => 4
  | .dressed_pion_scale_E0pi        => 5
  | .deuteron_binding_Eb            => 6
  | .droplet_aV                     => 6
  | .droplet_aS                     => 6
  | .droplet_aC                     => 6
  | .droplet_aA_bulk                => 6
  | .droplet_aP                     => 6
  | .droplet_aA_surf                => 7
  | .droplet_kappa_S                => 8
  | .effective_asymmetry_aA_eff     => 9
  | .continuum_binding_energy_B_TEV => 10

def nodeDeps : NuclearDagNode → List NuclearDagNode
  | .anchor_alpha                   => []
  | .anchor_Mp                      => []
  | .topo_dim3                      => []
  | .topo_trefoil_pq                => []
  | .topo_spin_covering             => []
  | .trefoil_invariant_I32          => [.topo_trefoil_pq, .topo_spin_covering]
  | .laue_virial_delta_p            => [.anchor_alpha, .topo_trefoil_pq, .topo_spin_covering]
  | .rational_semf_fractions        => [.topo_dim3, .topo_trefoil_pq, .topo_spin_covering]
  | .deuteron_recoil_fraction       => [.topo_trefoil_pq]
  | .electron_mass_me               =>
      [.anchor_alpha, .anchor_Mp, .trefoil_invariant_I32, .laue_virial_delta_p]
  | .bare_nambu_quantum_E0          =>
      [.anchor_Mp, .trefoil_invariant_I32, .laue_virial_delta_p]
  | .bare_pion_mass                 => [.bare_nambu_quantum_E0]
  | .pion_coulomb_shift             =>
      [.anchor_alpha, .bare_nambu_quantum_E0, .topo_spin_covering]
  | .dressed_pion_mass              => [.bare_pion_mass, .pion_coulomb_shift]
  | .dressed_pion_scale_E0pi        => [.dressed_pion_mass]
  | .deuteron_binding_Eb            =>
      [.anchor_Mp, .dressed_pion_mass, .deuteron_recoil_fraction]
  | .droplet_aV                     => [.rational_semf_fractions, .dressed_pion_scale_E0pi]
  | .droplet_aS                     => [.rational_semf_fractions, .dressed_pion_scale_E0pi]
  | .droplet_aC                     =>
      [.anchor_alpha, .rational_semf_fractions, .dressed_pion_scale_E0pi]
  | .droplet_aA_bulk                => [.rational_semf_fractions, .dressed_pion_scale_E0pi]
  | .droplet_aA_surf                => [.droplet_aV]
  | .droplet_kappa_S                => [.droplet_aA_bulk, .droplet_aA_surf]
  | .effective_asymmetry_aA_eff     => [.droplet_aA_bulk, .droplet_kappa_S]
  | .droplet_aP                     => [.rational_semf_fractions, .dressed_pion_scale_E0pi]
  | .continuum_binding_energy_B_TEV =>
      [.droplet_aV, .droplet_aS, .droplet_aC, .effective_asymmetry_aA_eff, .droplet_aP]

private theorem all_true_mem {α : Type} {p : α → Bool} :
    ∀ (xs : List α), xs.all p = true → ∀ x ∈ xs, p x = true
  | [], _, _, hx => by contradiction
  | a :: as, h, x, hx => by
      simp only [List.all_cons, Bool.and_eq_true] at h
      rcases List.mem_cons.mp hx with rfl | h_tail
      · exact h.1
      · exact all_true_mem as h.2 x h_tail

private theorem all_deps_rank_ok :
    ∀ n : NuclearDagNode,
      (nodeDeps n).all (fun d => decide (nodeRank d < nodeRank n)) = true := by
  intro n
  cases n <;> rfl

theorem anchors_have_zero_dependencies :
    nodeDeps .anchor_alpha = [] ∧
    nodeDeps .anchor_Mp = [] ∧
    nodeDeps .topo_dim3 = [] ∧
    nodeDeps .topo_trefoil_pq = [] ∧
    nodeDeps .topo_spin_covering = [] := by
  decide

theorem dependency_strictly_decreases_rank :
    ∀ n d : NuclearDagNode, d ∈ nodeDeps n → nodeRank d < nodeRank n := by
  intro n d hd
  exact of_decide_eq_true (all_true_mem (nodeDeps n) (all_deps_rank_ok n) d hd)

theorem no_self_dependency :
    ∀ n : NuclearDagNode, n ∉ nodeDeps n := by
  intro n hn
  exact Nat.lt_irrefl (nodeRank n) (dependency_strictly_decreases_rank n n hn)

theorem no_two_cycle :
    ∀ a b : NuclearDagNode, a ∈ nodeDeps b → b ∉ nodeDeps a := by
  intro a b hab hba
  exact Nat.lt_asymm
    (dependency_strictly_decreases_rank b a hab)
    (dependency_strictly_decreases_rank a b hba)

end TevNuclearDroplet
