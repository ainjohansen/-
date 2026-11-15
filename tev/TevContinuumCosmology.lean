import Mathlib
import tev.TevFoundations
import tev.TevCosmology
import tev.TevGravityInterface
import tev.TevFractionalLagrangian
import tev.TevRelationalPhase

set_option linter.style.whitespace false
set_option linter.style.emptyLine false
set_option linter.style.longLine false

/-!
# ТОПОЛОГИЧЕСКАЯ ЭЛАСТОДИНАМИКА ВАКУУМА (ТЭВ)
Модуль: TevContinuumCosmology (Трек 3.2 дорожной карты v6.5)

1. Эластодинамика утяжки Эшелби (ΔV_core < 0, u_r < 0):
   - точное вычисление ε_rr = du_r/dr и ε_θθ = u_r/r;
   - строгая бесследовость 3D-утяжки вне ядра: tr(ε) = ε_rr + 2·ε_θθ ≡ 0.
2. Эффективная метрика ОТО g_{00} = 1 + 2·ε_{00} = 1 - 2 G_N M / (c² r):
   - сохранение временно́й сигнатуры вне радиуса Шварцшильда r > r_s;
   - согласование с оптико-акустическим показателем преломления n_vac до O(x²).
3. Дробный лапласиан (-Δ)^γ в ИК-пределе и MOND без тёмной материи:
   - точное решение уравнения Милгрома и универсальные двусторонние оценки;
   - строгий асимптотический предел плоских кривых вращения v⁴(r) → G_N M a₀ при r → ∞.
4. Конечность резонатора Хаббла:
   - инфракрасная масса фотона m_γ = ħ H₀ / c² и её комптоновский масштаб R_H = c / H₀;
   - диссипативный дрейф вязкоупругости и статус текущих параметров равновесия матрицы.
-/

namespace TevContinuumCosmology

noncomputable section

open Filter

-- ============================================================================
-- 1. Утяжка Эшелби, бесследовость деформаций и эффективная метрика ОТО
-- ============================================================================

/-- Радиальное смещение сжимающего включения Эшелби (ΔV_core < 0, C > 0):
    u_r(r) = - C / r² (матрица утягивается внутрь к ядру дефекта) -/
def eshelbyRadialDisplacement (C r : ℝ) : ℝ :=
  - C / r ^ 2

-- ТЕОРЕМА 1: Поле смещений при утяжке строго направлено к центру (u_r < 0)
theorem eshelby_pull_strictly_inward (C r : ℝ) (hC : 0 < C) (hr : 0 < r) :
    eshelbyRadialDisplacement C r < 0 := by
  dsimp [eshelbyRadialDisplacement]
  have hr2 : 0 < r ^ 2 := sq_pos_of_pos hr
  have h_pos : 0 < C / r ^ 2 := div_pos hC hr2
  rw [neg_div]
  linarith

-- ТЕОРЕМА 2: Точная радиальная деформация ε_rr = du_r/dr = 2 C / r³
theorem eshelby_radial_strain_deriv (C r : ℝ) (hr : r ≠ 0) :
    HasDerivAt (eshelbyRadialDisplacement C) (2 * C / r ^ 3) r := by
  have h_fn : eshelbyRadialDisplacement C = fun x : ℝ => - C * (x ^ 2)⁻¹ := by
    ext x
    dsimp [eshelbyRadialDisplacement]
    rw [div_eq_mul_inv]
  rw [h_fn]
  have hp := hasDerivAt_pow 2 r
  have hi := hp.inv (pow_ne_zero 2 hr)
  have hm := hi.const_mul (-C)
  have h_eq : (-C) * (- (((2 : ℕ) : ℝ) * r ^ (2 - 1)) / (r ^ 2) ^ 2) = 2 * C / r ^ 3 := by
    have hp1 : (2 : ℕ) - 1 = 1 := rfl
    rw [hp1, pow_one]
    push_cast
    calc
      (-C) * (- (2 * r) / (r ^ 2) ^ 2)
        = (r / r) * (2 * C / r ^ 3) := by ring
      _ = 1 * (2 * C / r ^ 3) := by rw [div_self hr]
      _ = 2 * C / r ^ 3 := by ring
  rw [h_eq] at hm
  exact hm

-- ТЕОРЕМА 3 (БЕССЛЕДОВОСТЬ 3D-УТЯЖКИ ВНЕ КАВЕРНЫ):
-- Тензор сжимающей утяжки сохраняет квазинесжимаемость континуума: ε_rr + 2·ε_θθ ≡ 0
theorem eshelby_strain_trace_zero (C r : ℝ) (hr : r ≠ 0) :
    let eps_rr := deriv (eshelbyRadialDisplacement C) r
    let eps_theta := eshelbyRadialDisplacement C r / r
    eps_rr + 2 * eps_theta = 0 := by
  intro eps_rr eps_theta
  dsimp [eps_rr, eps_theta]
  rw [(eshelby_radial_strain_deriv C r hr).deriv]
  dsimp [eshelbyRadialDisplacement]
  rw [div_div, show r ^ 2 * r = r ^ 3 by ring]
  ring

/-- Безразмерный ньютоновский потенциал утяжки: ε_{00}(r) = Φ / c² = - G_N M / (c² r) -/
def gravitationalStrainPotential (GN M c r : ℝ) : ℝ :=
  - (GN * M) / (c ^ 2 * r)

/-- Эффективная метрическая компонента g_{00} = η_{00} + 2·ε_{00} = 1 - 2 G_N M / (c² r) -/
def effectiveG00 (GN M c r : ℝ) : ℝ :=
  1 + 2 * gravitationalStrainPotential GN M c r

/-- Радиус упругого срыва (горизонт Шварцшильда): r_s = 2 G_N M / c² -/
def schwarzschildRadius (GN M c : ℝ) : ℝ :=
  2 * GN * M / c ^ 2

-- ТЕОРЕМА 4: Вне радиуса Шварцшильда метрика строго сохраняет временну́ю сигнатуру (0 < g₀₀ < 1)
theorem metric_timelike_outside_horizon (GN M c r : ℝ)
    (hGN : 0 < GN) (hM : 0 < M) (hc : 0 < c)
    (h_outside : schwarzschildRadius GN M c < r) :
    0 < effectiveG00 GN M c r ∧ effectiveG00 GN M c r < 1 := by
  have hc2 : 0 < c ^ 2 := sq_pos_of_pos hc
  have hrs_nonneg : 0 ≤ schwarzschildRadius GN M c := by
    dsimp [schwarzschildRadius]
    positivity
  have hr : 0 < r := lt_of_le_of_lt hrs_nonneg h_outside
  have hc2r : 0 < c ^ 2 * r := mul_pos hc2 hr
  have h_rs_lt : 2 * GN * M < c ^ 2 * r := by
    dsimp [schwarzschildRadius] at h_outside
    rw [mul_comm (c ^ 2) r]
    exact (div_lt_iff₀ hc2).mp h_outside
  dsimp [effectiveG00, gravitationalStrainPotential]
  have h_frac_pos : 0 < (GN * M) / (c ^ 2 * r) := div_pos (mul_pos hGN hM) hc2r
  have h_2frac_lt : 2 * (GN * M / (c ^ 2 * r)) < 1 := by
    rw [show 2 * (GN * M / (c ^ 2 * r)) = (2 * GN * M) / (c ^ 2 * r) by ring]
    exact (div_lt_one hc2r).mpr h_rs_lt
  rw [neg_div]
  constructor <;> linarith

-- ТЕОРЕМА 5 (СШИВКА С ОПТИКО-АКУСТИЧЕСКОЙ МЕТРИКОЙ `TevFoundations`):
-- Произведение эффективного g₀₀ на квадрат показателя преломления n_vac отличается от 1
-- строго на постньютоновскую поправку второго порядка O((G_N M / (c² r))²).
theorem metric_acoustic_refractive_agreement (GN M c r : ℝ) :
    let x := (GN * M) / (c ^ 2 * r)
    let n := TevFoundations.vacuumRefractiveIndex GN M c r
    1 - effectiveG00 GN M c r * n ^ 2 = x ^ 2 * (3 + 2 * x) := by
  intro x n
  dsimp [effectiveG00, gravitationalStrainPotential, TevFoundations.vacuumRefractiveIndex, x, n]
  ring

/-- Ньютоновская константа связи из 18-ступенчатого каскада `TevGravityInterface` -/
def newtonGFromCascade (α hbar c Mp : ℝ) : ℝ :=
  TevGravityInterface.alphaG_refined α * hbar * c / Mp ^ 2

theorem newtonG_cascade_pos (α hbar c Mp : ℝ)
    (hα_pos : 0 < α) (hα_small : (21 / 5 : ℝ) * (α / Real.pi) < 1)
    (hhb : 0 < hbar) (hc : 0 < c) (hMp : 0 < Mp) :
    0 < newtonGFromCascade α hbar c Mp := by
  dsimp [newtonGFromCascade]
  have haG := TevGravityInterface.alphaG_refined_positive α hα_pos hα_small
  exact div_pos (mul_pos (mul_pos haG hhb) hc) (sq_pos_of_pos hMp)

-- ============================================================================
-- 2. ИК-модификация гравитации (MOND) и плоские кривые вращения галактик
-- ============================================================================

/-- Точное гравитационное ускорение с квадратичной функцией интерполяции Милгрома
    μ(g / a₀) = (g / a₀) / √(1 + (g / a₀)²), дающей μ(g / a₀) · g = g_N -/
def mondAccelerationField (g_N a₀ : ℝ) : ℝ :=
  Real.sqrt ((g_N ^ 2 + Real.sqrt (g_N ^ 4 + 4 * g_N ^ 2 * a₀ ^ 2)) / 2)

-- ТЕОРЕМА 6 (ТОЧНОЕ РЕШЕНИЕ УРАВНЕНИЯ МИЛГРОМА):
-- Определённое выше поле g = mondAccelerationField g_N a₀ является точным
-- положительным решением уравнения g² / √(g² + a₀²) = g_N.
theorem mond_interpolation_exact (g_N a₀ : ℝ) (hgN : 0 < g_N) (ha₀ : 0 < a₀) :
    let g := mondAccelerationField g_N a₀
    g ^ 2 / Real.sqrt (g ^ 2 + a₀ ^ 2) = g_N := by
  intro g
  set S := Real.sqrt (g_N ^ 4 + 4 * g_N ^ 2 * a₀ ^ 2)
  have h_rad_nonneg : 0 ≤ g_N ^ 4 + 4 * g_N ^ 2 * a₀ ^ 2 := by positivity
  have hS_sq : S ^ 2 = g_N ^ 4 + 4 * g_N ^ 2 * a₀ ^ 2 := Real.sq_sqrt h_rad_nonneg
  have hS_nonneg : 0 ≤ S := Real.sqrt_nonneg _
  set X := (g_N ^ 2 + S) / 2
  have hX_pos : 0 < X := by
    dsimp [X]
    have hgn2 : 0 < g_N ^ 2 := sq_pos_of_pos hgN
    linarith
  have hg_sq : g ^ 2 = X := by
    dsimp [g, mondAccelerationField]
    exact Real.sq_sqrt (le_of_lt hX_pos)
  rw [hg_sq]
  have h_X_eq : X ^ 2 = g_N ^ 2 * (X + a₀ ^ 2) := by
    calc
      X ^ 2 = (g_N ^ 4 + 2 * g_N ^ 2 * S + S ^ 2) / 4 := by
          dsimp [X]
          ring
      _ = (g_N ^ 4 + 2 * g_N ^ 2 * S + (g_N ^ 4 + 4 * g_N ^ 2 * a₀ ^ 2)) / 4 := by
          rw [hS_sq]
      _ = g_N ^ 2 * (X + a₀ ^ 2) := by
          dsimp [X]
          ring
  have h_den_pos : 0 < X + a₀ ^ 2 := by positivity
  have h_sq_eq : (X / Real.sqrt (X + a₀ ^ 2)) ^ 2 = g_N ^ 2 := by
    rw [div_pow, Real.sq_sqrt (le_of_lt h_den_pos), h_X_eq]
    exact mul_div_cancel_right₀ (g_N ^ 2) (ne_of_gt h_den_pos)
  have h_root := congrArg Real.sqrt h_sq_eq
  rwa [Real.sqrt_sq (div_nonneg (le_of_lt hX_pos) (Real.sqrt_nonneg _)),
       Real.sqrt_sq (le_of_lt hgN)] at h_root

-- ТЕОРЕМА 7 (УНИВЕРСАЛЬНЫЕ ОЦЕНКИ ДЛЯ УФ- И ИК-РЕЖИМОВ MOND):
-- Для всех g_N, a₀ > 0 одновременно выполняются:
-- 1) УФ (Ньютон): g_N² < g² < g_N² + a₀²;
-- 2) ИК (MOND):   g_N · a₀ < g² < g_N · a₀ + g_N².
theorem mond_universal_regime_bounds (g_N a₀ : ℝ) (hgN : 0 < g_N) (ha₀ : 0 < a₀) :
    let g := mondAccelerationField g_N a₀
    (g_N ^ 2 < g ^ 2 ∧ g ^ 2 < g_N ^ 2 + a₀ ^ 2) ∧
    (g_N * a₀ < g ^ 2 ∧ g ^ 2 < g_N * a₀ + g_N ^ 2) := by
  intro g
  set S := Real.sqrt (g_N ^ 4 + 4 * g_N ^ 2 * a₀ ^ 2)
  have h_rad_nonneg : 0 ≤ g_N ^ 4 + 4 * g_N ^ 2 * a₀ ^ 2 := by positivity
  have h_g_sq : g ^ 2 = (g_N ^ 2 + S) / 2 := by
    dsimp [g, mondAccelerationField]
    have h_pos : 0 ≤ (g_N ^ 2 + S) / 2 := by
      have : 0 ≤ S := Real.sqrt_nonneg _
      positivity
    exact Real.sq_sqrt h_pos
  have h1 : g_N ^ 2 < S := by
    have h_lt : g_N ^ 4 < g_N ^ 4 + 4 * g_N ^ 2 * a₀ ^ 2 := by
      have h_pos : 0 < 4 * g_N ^ 2 * a₀ ^ 2 := by positivity
      linarith
    have h_sq := Real.sqrt_lt_sqrt (by positivity) h_lt
    have h_lhs : Real.sqrt (g_N ^ 4) = g_N ^ 2 := by
      rw [show g_N ^ 4 = (g_N ^ 2) ^ 2 by ring, Real.sqrt_sq (by positivity)]
    rwa [h_lhs] at h_sq
  have h2 : 2 * g_N * a₀ < S := by
    have h_lt : (2 * g_N * a₀) ^ 2 < g_N ^ 4 + 4 * g_N ^ 2 * a₀ ^ 2 := by
      have h4 : 0 < g_N ^ 4 := pow_pos hgN 4
      linarith [show (2 * g_N * a₀) ^ 2 = 4 * g_N ^ 2 * a₀ ^ 2 by ring]
    have h_sq := Real.sqrt_lt_sqrt (by positivity) h_lt
    rwa [Real.sqrt_sq (by positivity)] at h_sq
  have h3 : S < g_N ^ 2 + 2 * a₀ ^ 2 := by
    have h_lt : g_N ^ 4 + 4 * g_N ^ 2 * a₀ ^ 2 < (g_N ^ 2 + 2 * a₀ ^ 2) ^ 2 := by
      have h4 : 0 < 4 * a₀ ^ 4 := by positivity
      linarith [show (g_N ^ 2 + 2 * a₀ ^ 2) ^ 2 =
        g_N ^ 4 + 4 * g_N ^ 2 * a₀ ^ 2 + 4 * a₀ ^ 4 by ring]
    have h_sq := Real.sqrt_lt_sqrt h_rad_nonneg h_lt
    rwa [Real.sqrt_sq (by positivity)] at h_sq
  have h4 : S < g_N ^ 2 + 2 * g_N * a₀ := by
    have h_lt : g_N ^ 4 + 4 * g_N ^ 2 * a₀ ^ 2 < (g_N ^ 2 + 2 * g_N * a₀) ^ 2 := by
      have h4 : 0 < 4 * g_N ^ 3 * a₀ := by positivity
      linarith [show (g_N ^ 2 + 2 * g_N * a₀) ^ 2 =
        g_N ^ 4 + 4 * g_N ^ 3 * a₀ + 4 * g_N ^ 2 * a₀ ^ 2 by ring]
    have h_sq := Real.sqrt_lt_sqrt h_rad_nonneg h_lt
    rwa [Real.sqrt_sq (by positivity)] at h_sq
  rw [h_g_sq]
  refine ⟨⟨by linarith, by linarith⟩, ⟨by linarith, by linarith⟩⟩

-- ТЕОРЕМА 8 (ПЛОСКИЕ КРИВЫЕ ВРАЩЕНИЯ ГАЛАКТИК БЕЗ ТЁМНОЙ МАТЕРИИ):
-- Для круговой орбиты v²(r) = r · g_MOND(G_N M / r², a₀) четвёртая степень скорости v⁴(r)
-- при r → ∞ строго сходится к барионному пределу Талли — Фишера G_N · M · a₀.
theorem flat_rotation_curve_asymptotic_limit (GN M c H0 : ℝ)
    (hGN : 0 < GN) (hM : 0 < M) (hc : 0 < c) (hH : 0 < H0) :
    let a₀ := TevCosmology.mondAcceleration c H0
    Tendsto (fun r : ℝ => r ^ 2 * (mondAccelerationField (GN * M / r ^ 2) a₀) ^ 2)
      atTop (nhds (GN * M * a₀)) := by
  intro a₀
  have ha₀ : 0 < a₀ := TevCosmology.mond_acceleration_pos c H0 hc hH
  have h_inv : Tendsto (fun r : ℝ => (1 / r : ℝ)) atTop (nhds 0) :=
    tendsto_const_nhds.div_atTop tendsto_id
  have h_sq_inv : Tendsto (fun r : ℝ => (GN * M) ^ 2 * ((1 / r) * (1 / r)))
      atTop (nhds ((GN * M) ^ 2 * (0 * 0))) :=
    tendsto_const_nhds.mul (h_inv.mul h_inv)
  have h_upper_lim : Tendsto
      (fun r : ℝ => GN * M * a₀ + (GN * M) ^ 2 * ((1 / r) * (1 / r)))
      atTop (nhds (GN * M * a₀)) := by
    have h := (tendsto_const_nhds (x := GN * M * a₀)).add h_sq_inv
    have h_zero : GN * M * a₀ + (GN * M) ^ 2 * (0 * 0) = GN * M * a₀ := by ring
    rw [h_zero] at h
    exact h
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds h_upper_lim ?_ ?_
  · filter_upwards [eventually_gt_atTop (0 : ℝ)] with r hr
    have hr2_pos : 0 < r ^ 2 := sq_pos_of_pos hr
    have hgN_pos : 0 < GN * M / r ^ 2 := div_pos (mul_pos hGN hM) hr2_pos
    have h_low := (mond_universal_regime_bounds (GN * M / r ^ 2) a₀ hgN_pos ha₀).2.1
    have h_mul := mul_lt_mul_of_pos_left h_low hr2_pos
    have h_lhs : r ^ 2 * (GN * M / r ^ 2 * a₀) = GN * M * a₀ := by
      field_simp [ne_of_gt hr]
    linarith
  · filter_upwards [eventually_gt_atTop (0 : ℝ)] with r hr
    have hr2_pos : 0 < r ^ 2 := sq_pos_of_pos hr
    have hgN_pos : 0 < GN * M / r ^ 2 := div_pos (mul_pos hGN hM) hr2_pos
    have h_up := (mond_universal_regime_bounds (GN * M / r ^ 2) a₀ hgN_pos ha₀).2.2
    have h_mul := mul_lt_mul_of_pos_left h_up hr2_pos
    have h_rhs : r ^ 2 * (GN * M / r ^ 2 * a₀ + (GN * M / r ^ 2) ^ 2) =
        GN * M * a₀ + (GN * M) ^ 2 * ((1 / r) * (1 / r)) := by
      have hr2_ne : r ^ 2 ≠ 0 := ne_of_gt hr2_pos
      calc
        r ^ 2 * (GN * M / r ^ 2 * a₀ + (GN * M / r ^ 2) ^ 2)
          = (r ^ 2 / r ^ 2) * (GN * M * a₀) +
              (r ^ 2 / r ^ 2) * ((GN * M) ^ 2 * ((1 / r) * (1 / r))) := by ring
        _ = 1 * (GN * M * a₀) + 1 * ((GN * M) ^ 2 * ((1 / r) * (1 / r))) := by
              rw [div_self hr2_ne]
        _ = GN * M * a₀ + (GN * M) ^ 2 * ((1 / r) * (1 / r)) := by ring
    linarith

-- ТЕОРЕМА 9 (ИНФРАКРАСНЫЙ КРОССОВЕР ДРОБНОГО ЛАПЛАСИАНА `TevFractionalLagrangian`):
-- В УФ (γ = 1) оператор (-Δ)¹ ведёт себя как ‖k‖² (3D-закон Ньютона),
-- а в ИК-пределе сферы Хаббла (γ = 1/2) переходит в линейный масштаб ‖k‖.
theorem ir_fractional_laplacian_crossover (k : TevFractionalLagrangian.WaveVector) :
    TevFractionalLagrangian.fracLaplacianSymbol 1 k = ‖k‖ ^ 2 ∧
    TevFractionalLagrangian.fracLaplacianSymbol (1 / 2) k = ‖k‖ := by
  dsimp [TevFractionalLagrangian.fracLaplacianSymbol]
  constructor
  · have h2 : (2 * (1 : ℝ)) = ((2 : ℕ) : ℝ) := by norm_num
    rw [h2, Real.rpow_natCast]
  · have h1 : (2 * (1 / 2 : ℝ)) = 1 := by norm_num
    rw [h1, Real.rpow_one]

-- ============================================================================
-- 3. Конечность резонатора Хаббла, масса фотона и текущий баланс матрицы
-- ============================================================================

/-- Радиус горизонта Хаббла (конечный размер резонатора Вселенной): R_H = c / H₀ -/
def hubbleRadius (c H0 : ℝ) : ℝ :=
  c / H0

/-- Эффективная инфракрасная масса фотона из конечности резонатора Хаббла:
    m_γ = ħ · H₀ / c² -/
def photonIRMass (hbar H0 c : ℝ) : ℝ :=
  hbar * H0 / c ^ 2

-- ТЕОРЕМА 10 (НЕНУЛЕВАЯ ИК-МАССА ФОТОНА И СОВПАДЕНИЕ С МАСШТАБОМ ХАББЛА):
-- Комптоновская длина волны ИК-массы фотона ħ / (m_γ c) тождественно равна R_H = c / H₀.
theorem photon_ir_mass_positive_and_compton_hubble (hbar H0 c : ℝ)
    (hhb : 0 < hbar) (hH : 0 < H0) (hc : 0 < c) :
    0 < photonIRMass hbar H0 c ∧
    hbar / (photonIRMass hbar H0 c * c) = hubbleRadius c H0 := by
  have hm_pos : 0 < photonIRMass hbar H0 c := by
    dsimp [photonIRMass]
    exact div_pos (mul_pos hhb hH) (sq_pos_of_pos hc)
  refine ⟨hm_pos, ?_⟩
  dsimp [photonIRMass, hubbleRadius]
  have hhb_ne : hbar ≠ 0 := ne_of_gt hhb
  have hH_ne : H0 ≠ 0 := ne_of_gt hH
  have hc_ne : c ≠ 0 := ne_of_gt hc
  field_simp [hhb_ne, hH_ne, hc_ne]

-- ТЕОРЕМА 11 (ДИССИПАТИВНОЕ ЗАТУХАНИЕ СДВИГОВЫХ ВОЛН НА КОСМОЛОГИЧЕСКОЙ БАЗЕ):
-- При вязкоупругом показателе матрицы 0 < ε_cos < 1 сдвиговая волна (фотон)
-- испытывает строго причинное ослабление, задаваемое шириной Γ(ε_cos) = Ω · tan(πε_cos/2).
theorem hubble_dissipative_redshift_bridge (ω Ω_k ε_cos : ℝ)
    (hω : 0 < ω) (hΩ : 0 < Ω_k) (hε_pos : 0 < ε_cos) (hε_lt : ε_cos < 1) :
    0 < (TevFractionalLagrangian.fractionalTimeSymbol ω ε_cos).im ∧
    TevRelationalPhase.resonanceDecayWidth Ω_k
      (TevFractionalLagrangian.fractionalQualityFactor ε_cos) =
      Ω_k * Real.tan (Real.pi * ε_cos / 2) := by
  exact ⟨(TevFractionalLagrangian.arrow_of_time_strict_dissipation ω ε_cos hω hε_pos hε_lt).1,
         (TevRelationalPhase.fractional_dissipation_sets_pole_width Ω_k ε_cos hΩ hε_pos hε_lt).2.2⟩

/-- Состояние текущего равновесного баланса матрицы (мировые константы как параметры эпохи):
    фиксирует упругую плотность ρ₀, модули G₀ ≪ K₀, безразмерную податливость α
    и текущий параметр Хаббла H₀, задающий ИК-рубеж a₀ и массу фотона m_γ. -/
structure CurrentMatrixEquilibrium where
  α : ℝ
  G₀ : ℝ
  K₀ : ℝ
  ρ₀ : ℝ
  H₀ : ℝ
  hbar : ℝ
  hα : 0 < α
  hG : 0 < G₀
  hK : G₀ < K₀
  hρ : 0 < ρ₀
  hH : 0 < H₀
  hhbar : 0 < hbar

/-- Скорость поперечных сдвиговых волн (света) в текущем равновесии матрицы: c = √(G₀ / ρ₀) -/
def equilibriumWaveSpeed (eq : CurrentMatrixEquilibrium) : ℝ :=
  Real.sqrt (eq.G₀ / eq.ρ₀)

theorem equilibrium_wave_speed_pos (eq : CurrentMatrixEquilibrium) :
    0 < equilibriumWaveSpeed eq :=
  Real.sqrt_pos.mpr (div_pos eq.hG eq.hρ)

theorem equilibrium_mond_and_photon_pos (eq : CurrentMatrixEquilibrium) :
    let c := equilibriumWaveSpeed eq
    0 < TevCosmology.mondAcceleration c eq.H₀ ∧
    0 < photonIRMass eq.hbar eq.H₀ c := by
  intro c
  have hc : 0 < c := equilibrium_wave_speed_pos eq
  exact ⟨TevCosmology.mond_acceleration_pos c eq.H₀ hc eq.hH,
         (photon_ir_mass_positive_and_compton_hubble eq.hbar eq.H₀ c eq.hhbar eq.hH hc).1⟩

end

end TevContinuumCosmology
