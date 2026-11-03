# Полный отчет аудита графа ТЭВ v5.5 (SSOT)

**Всего узлов:** 268 | **Ацикличность:** 0 циклов | **Размерности:** 268 PASS, 0 FAIL | **Расчеты:** 156 MATCH, 19 EXACT_ZERO, 0 MISMATCH

## ✓ Подтвержденные топологические нули и законы сохранения (EXACT_ZERO: 19)

| ID узла | Глава | Физический смысл | Lean 4 Теорема |
| :--- | :--- | :--- | :--- |
| `CONT_N_longitudinal` | 01_ontology | `0` | `-` |
| `CONT_solenoidal_div_u` | 01_ontology | `0` | `solenoidal_smul` |
| `CONT_laplace_pressure` | 01_ontology | `0` | `-` |
| `EM_div_B_zero` | 02_electromagnetism | `0` | `div_curl_identically_zero` |
| `EM_div_E_bulk` | 02_electromagnetism | `0` | `-` |
| `DEF_photon_mass` | 03_dimensions | `0` | `-` |
| `DEF_neutrino_charge` | 03_dimensions | `0` | `-` |
| `CLIFF_mean_curvature_zero` | 05_clifford_proton | `(1 - 1) / 2` | `-` |
| `BAR_neutron_charge_zero` | 07_baryons | `1 - 1` | `neutron_charge_is_zero` |
| `NEU_charge_zero` | 08_neutrinos | `0` | `-` |
| `NEU_m_betabeta` | 08_neutrinos | `0` | `majorana_mass_identically_zero` |
| `GRAV_laue_integral` | 11_gravity | `0` | `-` |
| `GRAV_tridiagonal_chain` | 11_gravity | `0` | `-` |
| `COSMO_lambda_zero` | 11_gravity | `0` | `-` |
| `STAT_pauli_zero` | 12_quantum | `0` | `pauli_exclusion_principle` |
| `LEP_electron_tension` | 06_leptons | `1 - 1` | `electron_is_ground_state` |
| `BAR_radiation_cancellation` | 07_baryons | `0` | `three_phase_signal_cancels` |
| `BAR_P_rad_zero` | 07_baryons | `0` | `-` |
| `GRAV_strain_trace_zero` | 11_gravity | `0` | `-` |

## ✓ Успешно подтвержденные численные значения (MATCH: 156)

| ID узла | Глава | Размерность | Вычислено (50 dps) | Заявлено | ppm | Lean 4 Теорема |
| :--- | :--- | :---: | :--- | :--- | :---: | :--- |
| `CONT_spinor_order` | 01_ontology | `[0, 0, 0, 0]` | 1.000000e+00 | 1 | 0.00 | `-` |
| `ROOT_alpha` | 01_ontology | `[0, 0, 0, 0]` | 7.297353e-03 | 0.0072973525643 | 0.00 | `-` |
| `SCALE_lambda_p` | 01_ontology | `[0, 1, 0, 0]` | 2.103089e-01 | 0.21030891 | 0.00 | `-` |
| `EM_charge_integral` | 02_electromagnetism | `[0, 0, 1, 1]` | 1.602177e-19 | 1.602176634e-19 | 0.00 | `electron_charge_is_one` |
| `CONT_Z_0` | 02_mises | `[1, 2, -3, -2]` | 3.767303e+02 | 376.730313668 | 0.00 | `-` |
| `CONT_R_K` | 02_mises | `[1, 2, -3, -2]` | 2.581281e+04 | 25812.8074593 | 0.00 | `-` |
| `TOP_I_trefoil` | 05_clifford_proton | `[0, 0, 0, 0]` | 1.340000e+01 | 13.4 | 0.00 | `trefoil_energy` |
| `KNOT_competitor_bound` | 05_clifford_proton | `[0, 0, 0, 0]` | 2.500000e+01 | 25 | 0.00 | `competing_knots_base_ge_25` |
| `KNOT_4_3_energy` | 05_clifford_proton | `[0, 0, 0, 0]` | 2.528571e+01 | 25.285714 | 0.01 | `-` |
| `SCALE_M_lobe` | 06_leptons | `[1, 0, 0, 0]` | 3.127574e+02 | 312.75736314 | 0.00 | `-` |
| `LEP_theta_0` | 06_leptons | `[0, 0, 0, 0]` | 2.222222e-01 | 0.222222 | 1.00 | `lode_angle_exact` |
| `LEP_laue_enthalpy` | 06_leptons | `[0, 0, 0, 0]` | 1.333333e+00 | 1.333333 | 0.25 | `-` |
| `LEP_m_e_exp_benchmark` | 06_leptons | `[1, 0, 0, 0]` | 5.109989e-01 | 0.510998950 | 0.00 | `-` |
| `LEP_dirac_g_factor` | 06_leptons | `[0, 0, 0, 0]` | 2.000000e+00 | 2 | 0.00 | `dirac_g_factor_exact` |
| `LEP_C1_coeff` | 06_leptons | `[0, 0, 0, 0]` | 5.000000e-01 | 0.5 | 0.00 | `C1_exact` |
| `LEP_I_geom_val` | 06_leptons | `[0, 0, 0, 0]` | 1.368056e+00 | 1.368056 | 0.32 | `I_geom_exact` |
| `LEP_I_geom_basis` | 06_leptons | `[0, 0, 0, 0]` | 1.440000e+02 | 144 | 0.00 | `I_geom_exact` |
| `LEP_anomaly_ratio_val` | 06_leptons | `[0, 0, 0, 0]` | 2.800000e+00 | 2.8 | 0.00 | `lepton_anomaly_ratio_exact` |
| `BAR_phase_shift_step` | 07_baryons | `[0, 0, 0, 0]` | 2.094395e+00 | 2.094395 | 0.05 | `-` |
| `BAR_knot_pitch` | 07_baryons | `[0, 0, 0, 0]` | 6.666667e-01 | 0.666667 | 0.50 | `-` |
| `BAR_q0_monopole` | 07_baryons | `[0, 0, 1, 1]` | 3.333333e-01 | 0.333333 | 1.00 | `q_monopole_val` |
| `BAR_strangeness_max` | 07_baryons | `[0, 0, 0, 0]` | 3.000000e+00 | 3 | 0.00 | `max_strangeness_is_three` |
| `BAR_C_Lambda` | 07_baryons | `[0, 0, 0, 0]` | 1.666667e+00 | 1.666667 | 0.20 | `-` |
| `CKM_delta_CP` | 07_baryons | `[0, 0, 0, 0]` | 1.130286e+00 | 1.13029 | 3.84 | `tan_delta_CP_sq` |
| `NEU_circulation_quantum` | 08_neutrinos | `[0, 0, 0, 0]` | 6.283185e+00 | 6.283185 | 0.05 | `-` |
| `TOP_N_3D_calc` | 08_neutrinos | `[0, 0, 0, 0]` | 5.000000e+00 | 5 | 0.00 | `deviator_dim_is_five` |
| `TOP_N_1D` | 08_neutrinos | `[0, 0, 0, 0]` | 3.000000e+00 | 3 | 0.00 | `-` |
| `NEU_axial_energy_fraction` | 08_neutrinos | `[0, 0, 0, 0]` | 2.000000e-01 | 0.2 | 0.00 | `-` |
| `NEU_gauge_dim` | 08_neutrinos | `[0, 0, 0, 0]` | 1.200000e+01 | 12 | 0.00 | `-` |
| `NEU_sum_stability_grad` | 08_neutrinos | `[1, 0, 0, 0]` | 2.000000e-02 | 0.02 | 0.00 | `-` |
| `TOP_N_basis_ew` | 09_electroweak | `[0, 0, 0, 0]` | 1.300000e+01 | 13 | 0.00 | `basis_dim_exact` |
| `EW_sin2_thetaW` | 09_electroweak | `[0, 0, 0, 0]` | 2.307692e-01 | 0.230769 | 1.00 | `weinberg_sin2_exact` |
| `EW_C_Z_derived` | 09_electroweak | `[0, 0, 0, 0]` | 1.250000e+00 | 1.25 | 0.00 | `C_Z_operator_exact` |
| `EW_C_W_derived` | 09_electroweak | `[0, 0, 0, 0]` | 3.500000e+00 | 3.5 | 0.00 | `C_W_operator_exact` |
| `EW_C_H_derived` | 09_electroweak | `[0, 0, 0, 0]` | -7.800000e+00 | -7.8 | 0.00 | `C_H_operator_exact` |
| `QCD_strong_base_dim` | 10_nuclear | `[0, 0, 0, 0]` | 1.700000e+01 | 17 | 0.00 | `strong_base_is_17` |
| `NUC_Eb_deuteron` | 10_nuclear | `[1, 0, 0, 0]` | 2.224442e+00 | 2.224442 | 0.18 | `deuteron_recoil_exact` |
| `NUC_tetra_sum_angle` | 10_nuclear | `[0, 0, 0, 0]` | 3.526439e+02 | 352.6439 | 0.01 | `tetrahedron_frustration_inequality` |
| `NUC_angular_deficit` | 10_nuclear | `[0, 0, 0, 0]` | 1.283882e-01 | 0.128388 | 1.72 | `-` |
| `GRAV_N_cascade_val` | 11_gravity | `[0, 0, 0, 0]` | 1.800000e+01 | 18 | 0.00 | `cascade_total_is_18` |
| `GRAV_C_total_val` | 11_gravity | `[0, 0, 0, 0]` | 4.200000e+00 | 4.2 | 0.00 | `total_screening_coeff_exact` |
| `GRAV_lab_dispersion_table` | 11_gravity | `[-1, 3, -2, 0]` | 6.673950e-11 | 6.67391e-11 | 5.99 | `-` |
| `GW_speed_equality` | 11_gravity | `[0, 1, -1, 0]` | 2.997925e+08 | 299792458 | 0.00 | `-` |
| `QM_uncertainty_relation` | 12_quantum | `[1, 2, -1, 0]` | 5.272859e-35 | 0.527286e-34 | 0.17 | `-` |
| `STAT_fermion_exchange` | 12_quantum | `[0, 0, 0, 0]` | -1.000000e+00 | -1 | 0.00 | `fermion_spin_half_antisymmetric` |
| `STAT_look_elsewhere_p_val` | 16_conclusion | `[0, 0, 0, 0]` | 8.946699e-25 | 8.95e-25 | 368.81 | `-` |
| `APP_clifford_torus_metric` | 17_appendix_faq | `[0, 2, 0, 0]` | 5.000000e-01 | 0.5 | 0.00 | `-` |
| `APP_lode_angle_definition` | 17_appendix_faq | `[0, 0, 0, 0]` | 2.222222e-01 | 0.222222 | 1.00 | `-` |
| `GRAV_alpha_G` | 01_ontology | `[0, 0, 0, 0]` | 5.905801e-39 | 5.905792e-39 | 1.60 | `alphaG_refined` |
| `DEF_neutrino_scale_concept` | 03_dimensions | `[1, 0, 0, 0]` | 1.229177e+01 | 12.2918 | 2.24 | `-` |
| `SCALE_delta_geom` | 06_leptons | `[0, 0, 0, 0]` | 3.484229e-03 | 0.00348424 | 3.10 | `-` |
| `LEP_proton_defect` | 06_leptons | `[0, 0, 0, 0]` | -7.100181e-05 | -7.10023e-5 | 6.96 | `-` |
| `LEP_a_l_schwinger_val` | 06_leptons | `[0, 0, 0, 0]` | 1.161410e-03 | 0.00116141 | 0.23 | `-` |
| `BAR_delta_tau_purcell` | 07_baryons | `[0, 0, 1, 0]` | 8.637146e+00 | 8.637 | 16.96 | `tau_bottle_strictly_less` |
| `CKM_cabibbo_angle` | 07_baryons | `[0, 0, 0, 0]` | 2.268679e-01 | 0.226868 | 0.61 | `cabibbo_angle_def` |
| `NEU_M_scale` | 08_neutrinos | `[1, 0, 0, 0]` | 1.229177e+01 | 12.2918 | 2.24 | `-` |
| `NEU_theta_nu` | 08_neutrinos | `[0, 0, 0, 0]` | 2.471442e+00 | 2.471442 | 0.10 | `-` |
| `EW_E_EW` | 09_electroweak | `[1, 0, 0, 0]` | 1.285771e+02 | 128.577053 | 0.00 | `electroweakScale` |
| `MES_E_0` | 10_nuclear | `[1, 0, 0, 0]` | 7.002528e+01 | 70.025276 | 0.02 | `nambu_quantum_bridge` |
| `MES_E_0_series` | 10_nuclear | `[1, 0, 0, 0]` | 7.002528e+01 | 70.025276 | 0.01 | `-` |
| `GRAV_alpha_18_kernel` | 11_gravity | `[0, 0, 0, 0]` | 3.443308e-39 | 3.443308e-39 | 0.13 | `two_block_cascade_factorization` |
| `GRAV_alpha18_multiplicative` | 11_gravity | `[0, 0, 0, 0]` | 3.443308e-39 | 3.443308e-39 | 0.13 | `-` |
| `GRAV_delta_interface` | 11_gravity | `[0, 0, 0, 0]` | 2.787383e-03 | 0.00278738 | 1.20 | `interface_coupling_exact` |
| `GRAV_alpha_G_pred` | 11_gravity | `[0, 0, 0, 0]` | 5.905801e-39 | 5.905792e-39 | 1.60 | `alphaG_refined` |
| `GW_scalar_suppression` | 11_gravity | `[0, 0, 0, 0]` | 4.645639e-04 | 0.00046456 | 8.38 | `total_screening_coeff_exact` |
| `COSMO_baryon_asymmetry` | 11_gravity | `[0, 0, 0, 0]` | 6.076514e-10 | 6.0765e-10 | 2.38 | `baryon_asymmetry_pos` |
| `RED_proton_electron_dimensionless` | 16_conclusion | `[0, 0, 0, 0]` | 1.836152e+03 | 1836.152 | 0.01 | `-` |
| `RED_proton_planck_dimensionless` | 16_conclusion | `[0, 0, 0, 0]` | 7.684921e-20 | 7.685e-20 | 10.25 | `-` |
| `APP_alpha_G_jump_128` | 17_appendix_faq | `[0, 0, 0, 0]` | 3.413851e+00 | 3.413851 | 0.13 | `-` |
| `BAR_r_c` | 07_baryons | `[0, 1, 0, 0]` | 8.412356e-01 | 0.841236 | 0.43 | `charge_radius_exact` |
| `BAR_R_m` | 07_baryons | `[0, 1, 0, 0]` | 6.309267e-01 | 0.630927 | 0.43 | `mass_radius_exact` |
| `DEF_alpha_impedance` | 02_mises | `[0, 0, 0, 0]` | 7.297353e-03 | 0.00729735256 | 0.00 | `-` |
| `TWIST_bps_tunneling_ratio` | 02_mises | `[0, 0, 0, 0]` | 8.164966e-01 | 0.81649658 | 0.00 | `mises_bps_ratio_exact` |
| `HW_deviator_amplitude_sqrt2` | 02_mises | `[0, 0, 0, 0]` | 1.414214e+00 | 1.41421356 | 0.00 | `koide_amplitude_sqrt_two` |
| `KNOT_stability_gap` | 05_clifford_proton | `[0, 0, 0, 0]` | 1.188571e+01 | 11.885714 | 0.00 | `-` |
| `LEP_muon_tension` | 06_leptons | `[0, 0, 0, 0]` | 5.000000e-01 | 0.5 | 0.00 | `muon_spring_has_excess_tension` |
| `LEP_tau_tension` | 06_leptons | `[0, 0, 0, 0]` | 6.666667e-01 | 0.666667 | 0.50 | `tau_spring_has_highest_tension` |
| `LEP_m_tau_bare` | 06_leptons | `[1, 0, 0, 0]` | 1.770746e+03 | 1770.7465 | 0.00 | `-` |
| `LEP_m_mu_bare` | 06_leptons | `[1, 0, 0, 0]` | 1.052885e+02 | 105.2885 | 0.23 | `-` |
| `LEP_m_e_bare` | 06_leptons | `[1, 0, 0, 0]` | 5.092050e-01 | 0.509205 | 0.03 | `-` |
| `LEP_C2_analytical` | 06_leptons | `[0, 0, 0, 0]` | -3.284790e-01 | -0.328479 | 0.10 | `-` |
| `BAR_d_quark_charge` | 07_baryons | `[0, 0, 1, 1]` | -3.333333e-01 | -0.333333 | 1.00 | `-` |
| `BAR_delta_q_return` | 07_baryons | `[0, 0, 1, 1]` | 6.666667e-01 | 0.666667 | 0.50 | `dipole_return_shielding_val` |
| `BAR_q_eff_1` | 07_baryons | `[0, 0, 1, 1]` | 6.666667e-01 | 0.666667 | 0.50 | `q_eff_pos_val` |
| `BAR_q_eff_2` | 07_baryons | `[0, 0, 1, 1]` | 6.666667e-01 | 0.666667 | 0.50 | `q_eff_pos_val` |
| `NEU_A_1D_sq` | 08_neutrinos | `[0, 0, 0, 0]` | 1.200000e+00 | 1.2 | 0.00 | `amplitude_sq_exact` |
| `NEU_C2_coeff` | 08_neutrinos | `[0, 0, 0, 0]` | 9.000000e-01 | 0.9 | 0.00 | `neutrino_same_reduction_operator` |
| `NEU_sin2_2theta13` | 08_neutrinos | `[0, 0, 0, 0]` | 8.333333e-02 | 0.083333 | 4.00 | `reactor_angle_double_angle_identity` |
| `EW_cos2_thetaW` | 09_electroweak | `[0, 0, 0, 0]` | 7.692308e-01 | 0.769231 | 0.30 | `weinberg_cos2_exact` |
| `EW_sin2_thetaW_low_q` | 09_electroweak | `[0, 0, 0, 0]` | 2.385120e-01 | 0.2385 | 50.16 | `-` |
| `NUC_P_D_deuteron` | 10_nuclear | `[0, 0, 0, 0]` | 5.769231e-02 | 0.057692 | 5.33 | `deuteron_d_wave_exact` |
| `QCD_alpha_s_MZ_val` | 10_nuclear | `[0, 0, 0, 0]` | 1.180570e-01 | 0.118057 | 0.27 | `alpha_s_MZ_pos` |
| `NUC_Eb_deuteron_val` | 10_nuclear | `[1, 0, 0, 0]` | 2.224442e+00 | 2.224442 | 0.00 | `-` |
| `STAT_significance_sigma` | 16_conclusion | `[0, 0, 0, 0]` | 1.027000e+01 | 10.27 | 0.00 | `-` |
| `GRAV_G_N` | 01_ontology | `[-1, 3, -2, 0]` | 6.673907e-11 | 6.67395e-11 | 6.48 | `newton_gravitational_constant_bridge` |
| `LEP_mp_me_ratio` | 06_leptons | `[0, 0, 0, 0]` | 1.836152e+03 | 1836.151996 | 0.01 | `nambu_quantum_bridge` |
| `BAR_tau_bottle_val` | 07_baryons | `[0, 0, 1, 0]` | 8.790629e+02 | 879.06 | 3.25 | `-` |
| `CKM_V_us` | 07_baryons | `[0, 0, 0, 0]` | 2.249268e-01 | 0.224926 | 3.34 | `ckm_first_row_unitarity` |
| `CKM_V_ud` | 07_baryons | `[0, 0, 0, 0]` | 9.743757e-01 | 0.974376 | 0.33 | `ckm_first_row_unitarity` |
| `NEU_m1` | 08_neutrinos | `[1, 0, 0, 0]` | 2.460013e-01 | 0.246 | 5.15 | `-` |
| `NEU_m2` | 08_neutrinos | `[1, 0, 0, 0]` | 8.673782e+00 | 8.674 | 25.15 | `-` |
| `NEU_m3` | 08_neutrinos | `[1, 0, 0, 0]` | 5.008072e+01 | 50.081 | 5.50 | `-` |
| `EW_M_Z_bare` | 09_electroweak | `[1, 0, 0, 0]` | 9.091771e+01 | 90.9178 | 1.03 | `bareMassZ` |
| `EW_M_W_bare` | 09_electroweak | `[1, 0, 0, 0]` | 7.974010e+01 | 79.7402 | 1.21 | `bareMassW` |
| `EW_M_H_val` | 09_electroweak | `[1, 0, 0, 0]` | 1.262475e+02 | 126.248 | 4.01 | `-` |
| `EW_vev` | 09_electroweak | `[1, 0, 0, 0]` | 2.462063e+02 | 246.2063 | 0.05 | `vev_scale_ratio_exact` |
| `MES_M_rho_val` | 06_leptons | `[1, 0, 0, 0]` | 7.702780e+02 | 770.278 | 0.06 | `rho_mass_is_eleven_E0` |
| `MES_E_0_val` | 10_nuclear | `[1, 0, 0, 0]` | 7.002528e+01 | 70.025277 | 0.00 | `-` |
| `MES_M_pi_pure_theor` | 10_nuclear | `[1, 0, 0, 0]` | 1.400506e+02 | 140.05 | 3.96 | `pion_mass_is_two_E0` |
| `NUC_Eb_deuteron_tree` | 10_nuclear | `[1, 0, 0, 0]` | 2.239773e+00 | 2.2398 | 11.86 | `-` |
| `NUC_delta_pi` | 10_nuclear | `[1, 0, 0, 0]` | -5.086252e-01 | -0.508 | 1230.74 | `pion_defect_negative` |
| `NUC_semf_aV` | 10_nuclear | `[1, 0, 0, 0]` | 1.556117e+01 | 15.5612 | 1.76 | `semf_aV_exact` |
| `NUC_semf_aS` | 10_nuclear | `[1, 0, 0, 0]` | 1.750632e+01 | 17.5063 | 1.10 | `semf_aS_exact` |
| `NUC_semf_aC` | 10_nuclear | `[1, 0, 0, 0]` | 6.898488e-01 | 0.68985 | 1.69 | `semf_aC_exact` |
| `NUC_semf_aA` | 10_nuclear | `[1, 0, 0, 0]` | 2.334176e+01 | 23.3418 | 1.76 | `semf_aA_exact` |
| `NUC_semf_aP` | 10_nuclear | `[1, 0, 0, 0]` | 1.167088e+01 | 11.6709 | 1.76 | `semf_aP_exact` |
| `GRAV_G_N_pred_val` | 11_gravity | `[-1, 3, -2, 0]` | 6.673907e-11 | 6.67395e-11 | 6.48 | `-` |
| `GW_scalar_power_ratio` | 11_gravity | `[0, 0, 0, 0]` | 2.158196e-07 | 2.158e-7 | 90.87 | `-` |
| `BAR_form_factor` | 07_baryons | `[0, 0, 0, 0]` | 7.500000e-01 | 0.75 | 0.00 | `nucleon_form_factor_exact` |
| `LEP_Q_koide` | 02_mises | `[0, 0, 0, 0]` | 6.666667e-01 | 0.66666667 | 0.01 | `koide_exact_from_mises_balance` |
| `LEP_m_tau` | 06_leptons | `[1, 0, 0, 0]` | 1.776916e+03 | 1776.916 | 0.10 | `-` |
| `LEP_m_mu` | 06_leptons | `[1, 0, 0, 0]` | 1.056553e+02 | 105.6553 | 0.23 | `-` |
| `BAR_q_eff_3` | 07_baryons | `[0, 0, 1, 1]` | -3.333333e-01 | -0.333333 | 1.00 | `q_eff_neg_val` |
| `NEU_Q_nu` | 08_neutrinos | `[0, 0, 0, 0]` | 5.333333e-01 | 0.533333 | 0.63 | `koide_neutrino_derived` |
| `NEU_sin2_theta13` | 08_neutrinos | `[0, 0, 0, 0]` | 2.128645e-02 | 0.021286 | 20.96 | `reactor_angle_double_angle_identity` |
| `QM_tsirelson_bound` | 12_quantum | `[0, 0, 0, 0]` | 2.828427e+00 | 2.828427 | 0.04 | `tsirelson_bound_exact` |
| `CONT_mu_str` | 01_ontology | `[1, -1, 0, 0]` | 4.286585e+26 | 4.2865e+26 | 19.73 | `-` |
| `CONT_mu_str_T0` | 11_gravity | `[1, -1, 0, 0]` | 4.286585e+26 | 4.2865e+26 | 19.73 | `-` |
| `GRAV_alpha_G_def_ratio` | 11_gravity | `[0, 0, 0, 0]` | 5.905801e-39 | 5.90610e-39 | 50.55 | `-` |
| `LEP_m_e` | 06_leptons | `[1, 0, 0, 0]` | 5.109991e-01 | 0.51099914 | 0.01 | `nambu_quantum_bridge` |
| `LEP_m_e_val` | 06_leptons | `[1, 0, 0, 0]` | 5.109991e-01 | 0.51099914 | 0.00 | `-` |
| `CKM_V_cb` | 07_baryons | `[0, 0, 0, 0]` | 4.130823e-02 | 0.04131 | 42.83 | `V_cb_nonneg` |
| `NEU_dm21_sq` | 08_neutrinos | `[2, 0, 0, 0]` | 7.517397e-05 | 7.5174e-5 | 0.33 | `-` |
| `NEU_dm31_sq` | 08_neutrinos | `[2, 0, 0, 0]` | 2.508018e-03 | 2.50802e-3 | 0.61 | `-` |
| `NEU_sum_m` | 08_neutrinos | `[1, 0, 0, 0]` | 5.900051e-02 | 0.0590 | 8.61 | `-` |
| `NEU_canonical_table_sum` | 08_neutrinos | `[1, 0, 0, 0]` | 5.900051e-02 | 0.0590005 | 0.13 | `-` |
| `EW_rho_tree` | 09_electroweak | `[0, 0, 0, 0]` | 1.000000e+00 | 1.0 | 0.00 | `veltman_rho_tree_level` |
| `EW_G_F` | 09_electroweak | `[-2, 0, 0, 0]` | 1.166505e-05 | 1.166505e-5 | 0.07 | `-` |
| `MES_M_pi_charged` | 10_nuclear | `[1, 0, 0, 0]` | 1.395419e+02 | 139.543 | 7.68 | `-` |
| `NUC_semf_sum_fractions` | 10_nuclear | `[0, 0, 0, 0]` | 9.722222e-01 | 0.972222 | 0.23 | `semf_fractions_sum_exact` |
| `BAR_delta_M_np` | 07_baryons | `[1, 0, 0, 0]` | 1.286776e+00 | 1.28678 | 2.96 | `neutron_heavier_than_proton` |
| `LEP_a_mu_HVP` | 06_leptons | `[0, 0, 0, 0]` | 6.767486e-08 | 6.77e-8 | 371.29 | `a_mu_HVP_narrow_pos` |
| `BAR_M_Lambda0` | 07_baryons | `[1, 0, 0, 0]` | 1.114364e+03 | 1114.36 | 3.86 | `-` |
| `BAR_energy_stationary` | 07_baryons | `[0, 2, 0, 0]` | 1.500000e+00 | 1.5 | 0.00 | `three_phase_energy_constant` |
| `CONT_T_0` | 01_ontology | `[1, 1, -2, 0]` | 3.852590e+43 | 3.8524e+43 | 49.34 | `-` |
| `CONT_rho_0` | 01_ontology | `[1, -3, 0, 0]` | 9.691613e+57 | 9.6915e+57 | 11.63 | `-` |
| `CONT_rho_0_val` | 11_gravity | `[1, -3, 0, 0]` | 9.691613e+57 | 9.6915e+57 | 11.63 | `-` |
| `LEP_R_e_torus` | 06_leptons | `[0, 1, 0, 0]` | 3.861591e+02 | 386.16 | 2.27 | `-` |
| `LEP_a_mu_HVP_val` | 06_leptons | `[0, 0, 0, 0]` | 6.770000e-08 | 6.77e-8 | 0.00 | `-` |
| `CONT_G_0` | 01_ontology | `[1, -1, -2, 0]` | 8.710387e+74 | 8.710e+74 | 44.45 | `-` |
| `LEP_r_e_core` | 06_leptons | `[0, 1, 0, 0]` | 2.817939e+00 | 2.818 | 21.55 | `electron_core_is_classical_radius` |
| `CONT_sigma_yield` | 01_ontology | `[1, -1, -2, 0]` | 6.356277e+72 | 6.356e72 | 43.52 | `-` |
| `SI_c_derived` | 01_ontology | `[0, 1, -1, 0]` | 2.997925e+08 | 299792458 | 0.00 | `-` |
| `CONT_tau_c_yield` | 02_mises | `[1, -1, -2, 0]` | 6.356277e+72 | 6.356e72 | 43.52 | `-` |
| `CONT_frenkel_yield_proof` | 01_ontology | `[1, -1, -2, 0]` | 6.356277e+72 | 6.356e72 | 43.52 | `-` |
| `CONT_sigma_eq_exact` | 02_mises | `[1, -1, -2, 0]` | 1.100939e+73 | 1.1009e73 | 35.79 | `mises_equivalent_stress_ratio` |
