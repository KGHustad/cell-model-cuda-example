#include <chrono>

#include <math.h>
#include <stdio.h>
#include <string.h>

extern "C" {
#include "cli.h"
}

#include <cuda_runtime.h>

#define checkCuda(error) __checkCuda(error, __FILE__, __LINE__)

static inline void __checkCuda(cudaError_t error, const char *file, const int line)
{
    if (error != cudaSuccess) {
        printf("checkCuda error at %s:%i: %s\n", file, line,
               cudaGetErrorString(cudaGetLastError()));
        exit(EXIT_FAILURE);
    }
    return;
}

enum state {
    STATE_Xr1,
    STATE_Xr2,
    STATE_Xs,
    STATE_m,
    STATE_h,
    STATE_j,
    STATE_d,
    STATE_f,
    STATE_f2,
    STATE_fCass,
    STATE_s,
    STATE_r,
    STATE_Ca_i,
    STATE_R_prime,
    STATE_Ca_SR,
    STATE_Ca_ss,
    STATE_Na_i,
    STATE_V,
    STATE_K_i,
    NUM_STATES,
};

enum parameter {
    PARAM_celltype,
    PARAM_P_kna,
    PARAM_g_K1,
    PARAM_g_Kr,
    PARAM_g_Ks,
    PARAM_g_Na,
    PARAM_g_bna,
    PARAM_g_CaL,
    PARAM_i_CaL_lim_delta,
    PARAM_g_bca,
    PARAM_g_to,
    PARAM_K_mNa,
    PARAM_K_mk,
    PARAM_P_NaK,
    PARAM_K_NaCa,
    PARAM_K_sat,
    PARAM_Km_Ca,
    PARAM_Km_Nai,
    PARAM_alpha,
    PARAM_gamma,
    PARAM_K_pCa,
    PARAM_g_pCa,
    PARAM_g_pK,
    PARAM_Buf_c,
    PARAM_Buf_sr,
    PARAM_Buf_ss,
    PARAM_Ca_o,
    PARAM_EC,
    PARAM_K_buf_c,
    PARAM_K_buf_sr,
    PARAM_K_buf_ss,
    PARAM_K_up,
    PARAM_V_leak,
    PARAM_V_rel,
    PARAM_V_sr,
    PARAM_V_ss,
    PARAM_V_xfer,
    PARAM_Vmax_up,
    PARAM_k1_prime,
    PARAM_k2_prime,
    PARAM_k3,
    PARAM_k4,
    PARAM_max_sr,
    PARAM_min_sr,
    PARAM_Na_o,
    PARAM_Cm,
    PARAM_F,
    PARAM_R,
    PARAM_T,
    PARAM_V_c,
    PARAM_is_stimulated,
    PARAM_stim_amplitude,
    PARAM_K_o,
    NUM_PARAMS,
};

// Init state values
__global__ void init_state_values(double *d_states, const unsigned int num_cells,
                                  const unsigned int padded_num_cells)
{
    const int thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    d_states[padded_num_cells * STATE_Xr1 + thread_ind] = 0.0165;
    d_states[padded_num_cells * STATE_Xr2 + thread_ind] = 0.473;
    d_states[padded_num_cells * STATE_Xs + thread_ind] = 0.0174;
    d_states[padded_num_cells * STATE_m + thread_ind] = 0.00165;
    d_states[padded_num_cells * STATE_h + thread_ind] = 0.749;
    d_states[padded_num_cells * STATE_j + thread_ind] = 0.6788;
    d_states[padded_num_cells * STATE_d + thread_ind] = 3.288e-05;
    d_states[padded_num_cells * STATE_f + thread_ind] = 0.7026;
    d_states[padded_num_cells * STATE_f2 + thread_ind] = 0.9526;
    d_states[padded_num_cells * STATE_fCass + thread_ind] = 0.9942;
    d_states[padded_num_cells * STATE_s + thread_ind] = 0.999998;
    d_states[padded_num_cells * STATE_r + thread_ind] = 2.347e-08;
    d_states[padded_num_cells * STATE_Ca_i + thread_ind] = 0.000153;
    d_states[padded_num_cells * STATE_R_prime + thread_ind] = 0.8978;
    d_states[padded_num_cells * STATE_Ca_SR + thread_ind] = 4.272;
    d_states[padded_num_cells * STATE_Ca_ss + thread_ind] = 0.00042;
    d_states[padded_num_cells * STATE_Na_i + thread_ind] = 10.132;
    d_states[padded_num_cells * STATE_V + thread_ind] = -85.423;
    d_states[padded_num_cells * STATE_K_i + thread_ind] = 138.52;
}

// Default parameter values
void init_parameters_values(double *h_parameters)
{
    h_parameters[PARAM_celltype] = 0.0;
    h_parameters[PARAM_P_kna] = 0.03;
    h_parameters[PARAM_g_K1] = 5.405;
    h_parameters[PARAM_g_Kr] = 0.153;
    h_parameters[PARAM_g_Ks] = 0.098;
    h_parameters[PARAM_g_Na] = 14.838;
    h_parameters[PARAM_g_bna] = 0.00029;
    h_parameters[PARAM_g_CaL] = 3.98e-05;
    h_parameters[PARAM_i_CaL_lim_delta] = 1e-07;
    h_parameters[PARAM_g_bca] = 0.000592;
    h_parameters[PARAM_g_to] = 0.294;
    h_parameters[PARAM_K_mNa] = 40.0;
    h_parameters[PARAM_K_mk] = 1.0;
    h_parameters[PARAM_P_NaK] = 2.724;
    h_parameters[PARAM_K_NaCa] = 1000.0;
    h_parameters[PARAM_K_sat] = 0.1;
    h_parameters[PARAM_Km_Ca] = 1.38;
    h_parameters[PARAM_Km_Nai] = 87.5;
    h_parameters[PARAM_alpha] = 2.5;
    h_parameters[PARAM_gamma] = 0.35;
    h_parameters[PARAM_K_pCa] = 0.0005;
    h_parameters[PARAM_g_pCa] = 0.1238;
    h_parameters[PARAM_g_pK] = 0.0146;
    h_parameters[PARAM_Buf_c] = 0.2;
    h_parameters[PARAM_Buf_sr] = 10.0;
    h_parameters[PARAM_Buf_ss] = 0.4;
    h_parameters[PARAM_Ca_o] = 2.0;
    h_parameters[PARAM_EC] = 1.5;
    h_parameters[PARAM_K_buf_c] = 0.001;
    h_parameters[PARAM_K_buf_sr] = 0.3;
    h_parameters[PARAM_K_buf_ss] = 0.00025;
    h_parameters[PARAM_K_up] = 0.00025;
    h_parameters[PARAM_V_leak] = 0.00036;
    h_parameters[PARAM_V_rel] = 0.102;
    h_parameters[PARAM_V_sr] = 0.001094;
    h_parameters[PARAM_V_ss] = 5.468e-05;
    h_parameters[PARAM_V_xfer] = 0.0038;
    h_parameters[PARAM_Vmax_up] = 0.006375;
    h_parameters[PARAM_k1_prime] = 0.15;
    h_parameters[PARAM_k2_prime] = 0.045;
    h_parameters[PARAM_k3] = 0.06;
    h_parameters[PARAM_k4] = 0.005;
    h_parameters[PARAM_max_sr] = 2.5;
    h_parameters[PARAM_min_sr] = 1.0;
    h_parameters[PARAM_Na_o] = 140.0;
    h_parameters[PARAM_Cm] = 0.185;
    h_parameters[PARAM_F] = 96485.3415;
    h_parameters[PARAM_R] = 8314.472;
    h_parameters[PARAM_T] = 310.0;
    h_parameters[PARAM_V_c] = 0.016404;
    h_parameters[PARAM_is_stimulated] = 0.0;
    h_parameters[PARAM_stim_amplitude] = 52.0;
    h_parameters[PARAM_K_o] = 5.4;
}

// State index
int state_index(const char name[])
{
    if (strcmp(name, "Xr1") == 0) {
        return STATE_Xr1;
    } else if (strcmp(name, "Xr2") == 0) {
        return STATE_Xr2;
    } else if (strcmp(name, "Xs") == 0) {
        return STATE_Xs;
    } else if (strcmp(name, "m") == 0) {
        return STATE_m;
    } else if (strcmp(name, "h") == 0) {
        return STATE_h;
    } else if (strcmp(name, "j") == 0) {
        return STATE_j;
    } else if (strcmp(name, "d") == 0) {
        return STATE_d;
    } else if (strcmp(name, "f") == 0) {
        return STATE_f;
    } else if (strcmp(name, "f2") == 0) {
        return STATE_f2;
    } else if (strcmp(name, "fCass") == 0) {
        return STATE_fCass;
    } else if (strcmp(name, "s") == 0) {
        return STATE_s;
    } else if (strcmp(name, "r") == 0) {
        return STATE_r;
    } else if (strcmp(name, "Ca_i") == 0) {
        return STATE_Ca_i;
    } else if (strcmp(name, "R_prime") == 0) {
        return STATE_R_prime;
    } else if (strcmp(name, "Ca_SR") == 0) {
        return STATE_Ca_SR;
    } else if (strcmp(name, "Ca_ss") == 0) {
        return STATE_Ca_ss;
    } else if (strcmp(name, "Na_i") == 0) {
        return STATE_Na_i;
    } else if (strcmp(name, "V") == 0) {
        return STATE_V;
    } else if (strcmp(name, "K_i") == 0) {
        return STATE_K_i;
    }
    return -1;
}

// Parameter index
int parameter_index(const char name[])
{
    if (strcmp(name, "celltype") == 0) {
        return PARAM_celltype;
    } else if (strcmp(name, "P_kna") == 0) {
        return PARAM_P_kna;
    } else if (strcmp(name, "g_K1") == 0) {
        return PARAM_g_K1;
    } else if (strcmp(name, "g_Kr") == 0) {
        return PARAM_g_Kr;
    } else if (strcmp(name, "g_Ks") == 0) {
        return PARAM_g_Ks;
    } else if (strcmp(name, "g_Na") == 0) {
        return PARAM_g_Na;
    } else if (strcmp(name, "g_bna") == 0) {
        return PARAM_g_bna;
    } else if (strcmp(name, "g_CaL") == 0) {
        return PARAM_g_CaL;
    } else if (strcmp(name, "i_CaL_lim_delta") == 0) {
        return PARAM_i_CaL_lim_delta;
    } else if (strcmp(name, "g_bca") == 0) {
        return PARAM_g_bca;
    } else if (strcmp(name, "g_to") == 0) {
        return PARAM_g_to;
    } else if (strcmp(name, "K_mNa") == 0) {
        return PARAM_K_mNa;
    } else if (strcmp(name, "K_mk") == 0) {
        return PARAM_K_mk;
    } else if (strcmp(name, "P_NaK") == 0) {
        return PARAM_P_NaK;
    } else if (strcmp(name, "K_NaCa") == 0) {
        return PARAM_K_NaCa;
    } else if (strcmp(name, "K_sat") == 0) {
        return PARAM_K_sat;
    } else if (strcmp(name, "Km_Ca") == 0) {
        return PARAM_Km_Ca;
    } else if (strcmp(name, "Km_Nai") == 0) {
        return PARAM_Km_Nai;
    } else if (strcmp(name, "alpha") == 0) {
        return PARAM_alpha;
    } else if (strcmp(name, "gamma") == 0) {
        return PARAM_gamma;
    } else if (strcmp(name, "K_pCa") == 0) {
        return PARAM_K_pCa;
    } else if (strcmp(name, "g_pCa") == 0) {
        return PARAM_g_pCa;
    } else if (strcmp(name, "g_pK") == 0) {
        return PARAM_g_pK;
    } else if (strcmp(name, "Buf_c") == 0) {
        return PARAM_Buf_c;
    } else if (strcmp(name, "Buf_sr") == 0) {
        return PARAM_Buf_sr;
    } else if (strcmp(name, "Buf_ss") == 0) {
        return PARAM_Buf_ss;
    } else if (strcmp(name, "Ca_o") == 0) {
        return PARAM_Ca_o;
    } else if (strcmp(name, "EC") == 0) {
        return PARAM_EC;
    } else if (strcmp(name, "K_buf_c") == 0) {
        return PARAM_K_buf_c;
    } else if (strcmp(name, "K_buf_sr") == 0) {
        return PARAM_K_buf_sr;
    } else if (strcmp(name, "K_buf_ss") == 0) {
        return PARAM_K_buf_ss;
    } else if (strcmp(name, "K_up") == 0) {
        return PARAM_K_up;
    } else if (strcmp(name, "V_leak") == 0) {
        return PARAM_V_leak;
    } else if (strcmp(name, "V_rel") == 0) {
        return PARAM_V_rel;
    } else if (strcmp(name, "V_sr") == 0) {
        return PARAM_V_sr;
    } else if (strcmp(name, "V_ss") == 0) {
        return PARAM_V_ss;
    } else if (strcmp(name, "V_xfer") == 0) {
        return PARAM_V_xfer;
    } else if (strcmp(name, "Vmax_up") == 0) {
        return PARAM_Vmax_up;
    } else if (strcmp(name, "k1_prime") == 0) {
        return PARAM_k1_prime;
    } else if (strcmp(name, "k2_prime") == 0) {
        return PARAM_k2_prime;
    } else if (strcmp(name, "k3") == 0) {
        return PARAM_k3;
    } else if (strcmp(name, "k4") == 0) {
        return PARAM_k4;
    } else if (strcmp(name, "max_sr") == 0) {
        return PARAM_max_sr;
    } else if (strcmp(name, "min_sr") == 0) {
        return PARAM_min_sr;
    } else if (strcmp(name, "Na_o") == 0) {
        return PARAM_Na_o;
    } else if (strcmp(name, "Cm") == 0) {
        return PARAM_Cm;
    } else if (strcmp(name, "F") == 0) {
        return PARAM_F;
    } else if (strcmp(name, "R") == 0) {
        return PARAM_R;
    } else if (strcmp(name, "T") == 0) {
        return PARAM_T;
    } else if (strcmp(name, "V_c") == 0) {
        return PARAM_V_c;
    } else if (strcmp(name, "is_stimulated") == 0) {
        return PARAM_is_stimulated;
    } else if (strcmp(name, "stim_amplitude") == 0) {
        return PARAM_stim_amplitude;
    } else if (strcmp(name, "K_o") == 0) {
        return PARAM_K_o;
    }
    return -1;
}

// Compute a forward step using the explicit Euler scheme to the TP06 ODE
__global__ void step_FE(double *__restrict d_states, const double t, const double dt,
                        const double *__restrict d_parameters, const unsigned int num_cells,
                        const unsigned int padded_num_cells)
{
    const int thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= num_cells)
        return; // number of nodes exceeded;

    // Assign states
    const double Xr1 = d_states[padded_num_cells * STATE_Xr1 + thread_ind];
    const double Xr2 = d_states[padded_num_cells * STATE_Xr2 + thread_ind];
    const double Xs = d_states[padded_num_cells * STATE_Xs + thread_ind];
    const double m = d_states[padded_num_cells * STATE_m + thread_ind];
    const double h = d_states[padded_num_cells * STATE_h + thread_ind];
    const double j = d_states[padded_num_cells * STATE_j + thread_ind];
    const double d = d_states[padded_num_cells * STATE_d + thread_ind];
    const double f = d_states[padded_num_cells * STATE_f + thread_ind];
    const double f2 = d_states[padded_num_cells * STATE_f2 + thread_ind];
    const double fCass = d_states[padded_num_cells * STATE_fCass + thread_ind];
    const double s = d_states[padded_num_cells * STATE_s + thread_ind];
    const double r = d_states[padded_num_cells * STATE_r + thread_ind];
    const double Ca_i = d_states[padded_num_cells * STATE_Ca_i + thread_ind];
    const double R_prime = d_states[padded_num_cells * STATE_R_prime + thread_ind];
    const double Ca_SR = d_states[padded_num_cells * STATE_Ca_SR + thread_ind];
    const double Ca_ss = d_states[padded_num_cells * STATE_Ca_ss + thread_ind];
    const double Na_i = d_states[padded_num_cells * STATE_Na_i + thread_ind];
    const double V = d_states[padded_num_cells * STATE_V + thread_ind];
    const double K_i = d_states[padded_num_cells * STATE_K_i + thread_ind];

    // Assign parameters
    const double celltype = d_parameters[PARAM_celltype];
    const double P_kna = d_parameters[PARAM_P_kna];
    const double g_K1 = d_parameters[PARAM_g_K1];
    const double g_Kr = d_parameters[PARAM_g_Kr];
    const double g_Ks = d_parameters[PARAM_g_Ks];
    const double g_Na = d_parameters[PARAM_g_Na];
    const double g_bna = d_parameters[PARAM_g_bna];
    const double g_CaL = d_parameters[PARAM_g_CaL];
    const double i_CaL_lim_delta = d_parameters[PARAM_i_CaL_lim_delta];
    const double g_bca = d_parameters[PARAM_g_bca];
    const double g_to = d_parameters[PARAM_g_to];
    const double K_mNa = d_parameters[PARAM_K_mNa];
    const double K_mk = d_parameters[PARAM_K_mk];
    const double P_NaK = d_parameters[PARAM_P_NaK];
    const double K_NaCa = d_parameters[PARAM_K_NaCa];
    const double K_sat = d_parameters[PARAM_K_sat];
    const double Km_Ca = d_parameters[PARAM_Km_Ca];
    const double Km_Nai = d_parameters[PARAM_Km_Nai];
    const double alpha = d_parameters[PARAM_alpha];
    const double gamma = d_parameters[PARAM_gamma];
    const double K_pCa = d_parameters[PARAM_K_pCa];
    const double g_pCa = d_parameters[PARAM_g_pCa];
    const double g_pK = d_parameters[PARAM_g_pK];
    const double Buf_c = d_parameters[PARAM_Buf_c];
    const double Buf_sr = d_parameters[PARAM_Buf_sr];
    const double Buf_ss = d_parameters[PARAM_Buf_ss];
    const double Ca_o = d_parameters[PARAM_Ca_o];
    const double EC = d_parameters[PARAM_EC];
    const double K_buf_c = d_parameters[PARAM_K_buf_c];
    const double K_buf_sr = d_parameters[PARAM_K_buf_sr];
    const double K_buf_ss = d_parameters[PARAM_K_buf_ss];
    const double K_up = d_parameters[PARAM_K_up];
    const double V_leak = d_parameters[PARAM_V_leak];
    const double V_rel = d_parameters[PARAM_V_rel];
    const double V_sr = d_parameters[PARAM_V_sr];
    const double V_ss = d_parameters[PARAM_V_ss];
    const double V_xfer = d_parameters[PARAM_V_xfer];
    const double Vmax_up = d_parameters[PARAM_Vmax_up];
    const double k1_prime = d_parameters[PARAM_k1_prime];
    const double k2_prime = d_parameters[PARAM_k2_prime];
    const double k3 = d_parameters[PARAM_k3];
    const double k4 = d_parameters[PARAM_k4];
    const double max_sr = d_parameters[PARAM_max_sr];
    const double min_sr = d_parameters[PARAM_min_sr];
    const double Na_o = d_parameters[PARAM_Na_o];
    const double Cm = d_parameters[PARAM_Cm];
    const double F = d_parameters[PARAM_F];
    const double R = d_parameters[PARAM_R];
    const double T = d_parameters[PARAM_T];
    const double V_c = d_parameters[PARAM_V_c];
    const double is_stimulated = d_parameters[PARAM_is_stimulated];
    const double stim_amplitude = d_parameters[PARAM_stim_amplitude];
    const double K_o = d_parameters[PARAM_K_o];

    // Expressions for the Reversal potentials component
    const double E_Na = R * T * log(Na_o / Na_i) / F;
    const double E_K = R * T * log(K_o / K_i) / F;
    const double E_Ks = R * T * log((K_o + Na_o * P_kna) / (P_kna * Na_i + K_i)) / F;
    const double E_Ca = 0.5 * R * T * log(Ca_o / Ca_i) / F;

    // Expressions for the Inward rectifier potassium current component
    const double alpha_K1 = 0.1 / (1. + 6.14421235332821e-6 * exp(0.06 * V - 0.06 * E_K));
    const double beta_K1 = (0.367879441171442 * exp(0.1 * V - 0.1 * E_K)
                            + 3.06060402008027 * exp(0.0002 * V - 0.0002 * E_K))
                           / (1. + exp(0.5 * E_K - 0.5 * V));
    const double xK1_inf = alpha_K1 / (alpha_K1 + beta_K1);
    const double i_K1 = 0.430331482911935 * g_K1 * sqrt(K_o) * (-E_K + V) * xK1_inf;

    // Expressions for the Rapid time dependent potassium current component
    const double i_Kr = 0.430331482911935 * g_Kr * sqrt(K_o) * (-E_K + V) * Xr1 * Xr2;

    // Expressions for the Xr1 gate component
    const double xr1_inf = 1.0 / (1. + exp(-26. / 7. - V / 7.));
    const double alpha_xr1 = 450. / (1. + exp(-9. / 2. - V / 10.));
    const double beta_xr1 = 6. / (1. + exp(60. / 23. + 2. * V / 23.));
    const double tau_xr1 = alpha_xr1 * beta_xr1;
    const double dXr1_dt = (-Xr1 + xr1_inf) / tau_xr1;
    d_states[padded_num_cells * STATE_Xr1 + thread_ind] = dt * dXr1_dt + Xr1;

    // Expressions for the Xr2 gate component
    const double xr2_inf = 1.0 / (1. + exp(11. / 3. + V / 24.));
    const double alpha_xr2 = 3. / (1. + exp(-3. - V / 20.));
    const double beta_xr2 = 1.12 / (1. + exp(-3. + V / 20.));
    const double tau_xr2 = alpha_xr2 * beta_xr2;
    const double dXr2_dt = (-Xr2 + xr2_inf) / tau_xr2;
    d_states[padded_num_cells * STATE_Xr2 + thread_ind] = dt * dXr2_dt + Xr2;

    // Expressions for the Slow time dependent potassium current component
    const double i_Ks = g_Ks * (Xs * Xs) * (-E_Ks + V);

    // Expressions for the Xs gate component
    const double xs_inf = 1.0 / (1. + exp(-5. / 14. - V / 14.));
    const double alpha_xs = 1400. / sqrt(1. + exp(5. / 6. - V / 6.));
    const double beta_xs = 1.0 / (1. + exp(-7. / 3. + V / 15.));
    const double tau_xs = 80. + alpha_xs * beta_xs;
    const double dXs_dt = (-Xs + xs_inf) / tau_xs;
    d_states[padded_num_cells * STATE_Xs + thread_ind] = dt * dXs_dt + Xs;

    // Expressions for the Fast sodium current component
    const double i_Na = g_Na * (m * m * m) * (-E_Na + V) * h * j;

    // Expressions for the m gate component
    const double m_inf = 1.0
                         / ((1. + exp(-5686. / 903. - 100. * V / 903.))
                            * (1. + exp(-5686. / 903. - 100. * V / 903.)));
    const double alpha_m = 1.0 / (1. + exp(-12. - V / 5.));
    const double beta_m = 0.1 / (1. + exp(7. + V / 5.)) + 0.1 / (1. + exp(-1. / 4. + V / 200.));
    const double tau_m = alpha_m * beta_m;
    const double dm_dt = (-m + m_inf) / tau_m;
    d_states[padded_num_cells * STATE_m + thread_ind] = dt * dm_dt + m;

    // Expressions for the h gate component
    const double h_inf = 1.0
                         / ((1. + exp(7155. / 743. + 100. * V / 743.))
                            * (1. + exp(7155. / 743. + 100. * V / 743.)));
    const double alpha_h = (V < -40. ? 4.43126792958051e-7 * exp(-0.147058823529412 * V) : 0.);
    const double beta_h =
            (V < -40. ? 310000. * exp(0.3485 * V) + 2.7 * exp(0.079 * V)
                      : 0.77 / (0.13 + 0.0497581410839387 * exp(-0.0900900900900901 * V)));
    const double tau_h = 1.0 / (alpha_h + beta_h);
    const double dh_dt = (-h + h_inf) / tau_h;
    d_states[padded_num_cells * STATE_h + thread_ind] = dt * dh_dt + h;

    // Expressions for the j gate component
    const double j_inf = 1.0
                         / ((1. + exp(7155. / 743. + 100. * V / 743.))
                            * (1. + exp(7155. / 743. + 100. * V / 743.)));
    const double alpha_j =
            (V < -40. ? (37.78 + V) * (-25428. * exp(0.2444 * V) - 6.948e-6 * exp(-0.04391 * V))
                                / (1. + 50262745825.954 * exp(0.311 * V))
                      : 0.);
    const double beta_j =
            (V < -40. ? 0.02424 * exp(-0.01052 * V) / (1. + 0.00396086833990426 * exp(-0.1378 * V))
                      : 0.6 * exp(0.057 * V) / (1. + exp(-16. / 5. - V / 10.)));
    const double tau_j = 1.0 / (alpha_j + beta_j);
    const double dj_dt = (-j + j_inf) / tau_j;
    d_states[padded_num_cells * STATE_j + thread_ind] = dt * dj_dt + j;

    // Expressions for the Sodium background current component
    const double i_b_Na = g_bna * (-E_Na + V);

    // Expressions for the L_type Ca current component
    const double V_eff = -15. + V;
    const double i_CaL_factors = 4. * F * g_CaL
                                 * (-Ca_o + Ca_ss * exp(2. * F * V_eff / (R * T)) / 4.) * d * f * f2
                                 * fCass;
    const double i_CaL_fraction =
            (fabs(V_eff) < i_CaL_lim_delta
                     ? 0.5
                     : F * V_eff / (R * T * (expm1(2. * F * V_eff / (R * T)))));
    const double i_CaL = i_CaL_factors * i_CaL_fraction;

    // Expressions for the d gate component
    const double d_inf = 1.0 / (1. + exp(-16. / 15. - 2. * V / 15.));
    const double alpha_d = 0.25 + 1.4 / (1. + exp(-35. / 13. - V / 13.));
    const double beta_d = 1.4 / (1. + exp(1. + V / 5.));
    const double gamma_d = 1.0 / (1. + exp(5. / 2. - V / 20.));
    const double tau_d = alpha_d * beta_d + gamma_d;
    const double dd_dt = (-d + d_inf) / tau_d;
    d_states[padded_num_cells * STATE_d + thread_ind] = dt * dd_dt + d;

    // Expressions for the f gate component
    const double f_inf = 1.0 / (1. + exp(20. / 7. + V / 7.));
    const double tau_f = 20. + 180. / (1. + exp(3. + V / 10.))
                         + 200. / (1. + exp(13. / 10. - V / 10.))
                         + 1102.5 * exp(-((27. + V) * (27. + V)) / 225.);
    const double df_dt = (-f + f_inf) / tau_f;
    d_states[padded_num_cells * STATE_f + thread_ind] = dt * df_dt + f;

    // Expressions for the F2 gate component
    const double f2_inf = 0.33 + 0.67 / (1. + exp(5. + V / 7.));
    const double tau_f2 = 31. / (1. + exp(5. / 2. - V / 10.)) + 80. / (1. + exp(3. + V / 10.))
                          + 562. * exp(-((27. + V) * (27. + V)) / 240.);
    const double df2_dt = (-f2 + f2_inf) / tau_f2;
    d_states[padded_num_cells * STATE_f2 + thread_ind] = dt * df2_dt + f2;

    // Expressions for the FCass gate component
    const double fCass_inf = 0.4 + 0.6 / (1. + 400. * (Ca_ss * Ca_ss));
    const double tau_fCass = 2. + 80. / (1. + 400. * (Ca_ss * Ca_ss));
    const double dfCass_dt = (-fCass + fCass_inf) / tau_fCass;
    d_states[padded_num_cells * STATE_fCass + thread_ind] = dt * dfCass_dt + fCass;

    // Expressions for the Calcium background current component
    const double i_b_Ca = g_bca * (-E_Ca + V);

    // Expressions for the Transient outward current component
    const double i_to = g_to * (-E_K + V) * r * s;

    // Expressions for the s gate component
    const double s_inf =
            (celltype == 2. ? 1.0 / (1. + exp(28. / 5. + V / 5.)) : 1.0 / (1. + exp(4. + V / 5.)));
    const double tau_s = (celltype == 2. ? 8. + 1000. * exp(-((67. + V) * (67. + V)) / 1000.)
                                         : 3. + 5. / (1. + exp(-4. + V / 5.))
                                                   + 85. * exp(-((45. + V) * (45. + V)) / 320.));
    const double ds_dt = (-s + s_inf) / tau_s;
    d_states[padded_num_cells * STATE_s + thread_ind] = dt * ds_dt + s;

    // Expressions for the r gate component
    const double r_inf = 1.0 / (1. + exp(10. / 3. - V / 6.));
    const double tau_r = 0.8 + 9.5 * exp(-((40. + V) * (40. + V)) / 1800.);
    const double dr_dt = (-r + r_inf) / tau_r;
    d_states[padded_num_cells * STATE_r + thread_ind] = dt * dr_dt + r;

    // Expressions for the Sodium potassium pump current component
    const double i_NaK =
            K_o * P_NaK * Na_i
            / ((K_mNa + Na_i) * (K_mk + K_o)
               * (1. + 0.0353 * exp(-F * V / (R * T)) + 0.1245 * exp(-0.1 * F * V / (R * T))));

    // Expressions for the Sodium calcium exchanger current component
    const double i_NaCa =
            K_NaCa
            * (Ca_o * (Na_i * Na_i * Na_i) * exp(F * gamma * V / (R * T))
               - alpha * (Na_o * Na_o * Na_o) * Ca_i * exp(F * (-1. + gamma) * V / (R * T)))
            / ((1. + K_sat * exp(F * (-1. + gamma) * V / (R * T))) * (Ca_o + Km_Ca)
               * ((Km_Nai * Km_Nai * Km_Nai) + (Na_o * Na_o * Na_o)));

    // Expressions for the Calcium pump current component
    const double i_p_Ca = g_pCa * Ca_i / (K_pCa + Ca_i);

    // Expressions for the Potassium pump current component
    const double i_p_K = g_pK * (-E_K + V) / (1. + exp(1250. / 299. - 50. * V / 299.));

    // Expressions for the Calcium dynamics component
    const double i_up = Vmax_up / (1. + (K_up * K_up) / (Ca_i * Ca_i));
    const double i_leak = V_leak * (-Ca_i + Ca_SR);
    const double i_xfer = V_xfer * (-Ca_i + Ca_ss);
    const double kcasr = max_sr - (max_sr - min_sr) / (1. + (EC * EC) / (Ca_SR * Ca_SR));
    const double Ca_i_bufc = 1.0 / (1. + Buf_c * K_buf_c / ((K_buf_c + Ca_i) * (K_buf_c + Ca_i)));
    const double Ca_sr_bufsr =
            1.0 / (1. + Buf_sr * K_buf_sr / ((K_buf_sr + Ca_SR) * (K_buf_sr + Ca_SR)));
    const double Ca_ss_bufss =
            1.0 / (1. + Buf_ss * K_buf_ss / ((K_buf_ss + Ca_ss) * (K_buf_ss + Ca_ss)));
    const double dCa_i_dt = (V_sr * (-i_up + i_leak) / V_c
                             - Cm * (-2. * i_NaCa + i_b_Ca + i_p_Ca) / (2. * F * V_c) + i_xfer)
                            * Ca_i_bufc;
    d_states[padded_num_cells * STATE_Ca_i + thread_ind] = dt * dCa_i_dt + Ca_i;
    const double k1 = k1_prime / kcasr;
    const double k2 = k2_prime * kcasr;
    const double O = (Ca_ss * Ca_ss) * R_prime * k1 / (k3 + (Ca_ss * Ca_ss) * k1);
    const double dR_prime_dt = k4 * (1. - R_prime) - Ca_ss * R_prime * k2;
    d_states[padded_num_cells * STATE_R_prime + thread_ind] = dt * dR_prime_dt + R_prime;
    const double i_rel = V_rel * (-Ca_ss + Ca_SR) * O;
    const double dCa_SR_dt = (-i_leak - i_rel + i_up) * Ca_sr_bufsr;
    d_states[padded_num_cells * STATE_Ca_SR + thread_ind] = dt * dCa_SR_dt + Ca_SR;
    const double dCa_ss_dt =
            (V_sr * i_rel / V_ss - V_c * i_xfer / V_ss - Cm * i_CaL / (2. * F * V_ss))
            * Ca_ss_bufss;
    d_states[padded_num_cells * STATE_Ca_ss + thread_ind] = dt * dCa_ss_dt + Ca_ss;

    // Expressions for the Sodium dynamics component
    const double dNa_i_dt = Cm * (-i_Na - i_b_Na - 3. * i_NaCa - 3. * i_NaK) / (F * V_c);
    d_states[padded_num_cells * STATE_Na_i + thread_ind] = dt * dNa_i_dt + Na_i;

    // Expressions for the Membrane component
    const double i_Stim = (is_stimulated ? -stim_amplitude : 0.);
    const double dV_dt = -i_CaL - i_K1 - i_Kr - i_Ks - i_Na - i_NaCa - i_NaK - i_Stim - i_b_Ca
                         - i_b_Na - i_p_Ca - i_p_K - i_to;
    d_states[padded_num_cells * STATE_V + thread_ind] = dt * dV_dt + V;

    // Expressions for the Potassium dynamics component
    const double dK_i_dt =
            Cm * (-i_K1 - i_Kr - i_Ks - i_Stim - i_p_K - i_to + 2. * i_NaK) / (F * V_c);
    d_states[padded_num_cells * STATE_K_i + thread_ind] = dt * dK_i_dt + K_i;
}

// Compute a forward step using the rush larsen algorithm to the TP06 ODE
__global__ void step_RL(double *d_states, const double t, const double dt,
                        const double *d_parameters, const unsigned int num_cells,
                        const unsigned int padded_num_cells)
{
    const int thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= num_cells)
        return; // number of nodes exceeded;

    // Assign states
    const double Xr1 = d_states[padded_num_cells * STATE_Xr1 + thread_ind];
    const double Xr2 = d_states[padded_num_cells * STATE_Xr2 + thread_ind];
    const double Xs = d_states[padded_num_cells * STATE_Xs + thread_ind];
    const double m = d_states[padded_num_cells * STATE_m + thread_ind];
    const double h = d_states[padded_num_cells * STATE_h + thread_ind];
    const double j = d_states[padded_num_cells * STATE_j + thread_ind];
    const double d = d_states[padded_num_cells * STATE_d + thread_ind];
    const double f = d_states[padded_num_cells * STATE_f + thread_ind];
    const double f2 = d_states[padded_num_cells * STATE_f2 + thread_ind];
    const double fCass = d_states[padded_num_cells * STATE_fCass + thread_ind];
    const double s = d_states[padded_num_cells * STATE_s + thread_ind];
    const double r = d_states[padded_num_cells * STATE_r + thread_ind];
    const double Ca_i = d_states[padded_num_cells * STATE_Ca_i + thread_ind];
    const double R_prime = d_states[padded_num_cells * STATE_R_prime + thread_ind];
    const double Ca_SR = d_states[padded_num_cells * STATE_Ca_SR + thread_ind];
    const double Ca_ss = d_states[padded_num_cells * STATE_Ca_ss + thread_ind];
    const double Na_i = d_states[padded_num_cells * STATE_Na_i + thread_ind];
    const double V = d_states[padded_num_cells * STATE_V + thread_ind];
    const double K_i = d_states[padded_num_cells * STATE_K_i + thread_ind];

    // Assign parameters
    const double celltype = d_parameters[PARAM_celltype];
    const double P_kna = d_parameters[PARAM_P_kna];
    const double g_K1 = d_parameters[PARAM_g_K1];
    const double g_Kr = d_parameters[PARAM_g_Kr];
    const double g_Ks = d_parameters[PARAM_g_Ks];
    const double g_Na = d_parameters[PARAM_g_Na];
    const double g_bna = d_parameters[PARAM_g_bna];
    const double g_CaL = d_parameters[PARAM_g_CaL];
    const double i_CaL_lim_delta = d_parameters[PARAM_i_CaL_lim_delta];
    const double g_bca = d_parameters[PARAM_g_bca];
    const double g_to = d_parameters[PARAM_g_to];
    const double K_mNa = d_parameters[PARAM_K_mNa];
    const double K_mk = d_parameters[PARAM_K_mk];
    const double P_NaK = d_parameters[PARAM_P_NaK];
    const double K_NaCa = d_parameters[PARAM_K_NaCa];
    const double K_sat = d_parameters[PARAM_K_sat];
    const double Km_Ca = d_parameters[PARAM_Km_Ca];
    const double Km_Nai = d_parameters[PARAM_Km_Nai];
    const double alpha = d_parameters[PARAM_alpha];
    const double gamma = d_parameters[PARAM_gamma];
    const double K_pCa = d_parameters[PARAM_K_pCa];
    const double g_pCa = d_parameters[PARAM_g_pCa];
    const double g_pK = d_parameters[PARAM_g_pK];
    const double Buf_c = d_parameters[PARAM_Buf_c];
    const double Buf_sr = d_parameters[PARAM_Buf_sr];
    const double Buf_ss = d_parameters[PARAM_Buf_ss];
    const double Ca_o = d_parameters[PARAM_Ca_o];
    const double EC = d_parameters[PARAM_EC];
    const double K_buf_c = d_parameters[PARAM_K_buf_c];
    const double K_buf_sr = d_parameters[PARAM_K_buf_sr];
    const double K_buf_ss = d_parameters[PARAM_K_buf_ss];
    const double K_up = d_parameters[PARAM_K_up];
    const double V_leak = d_parameters[PARAM_V_leak];
    const double V_rel = d_parameters[PARAM_V_rel];
    const double V_sr = d_parameters[PARAM_V_sr];
    const double V_ss = d_parameters[PARAM_V_ss];
    const double V_xfer = d_parameters[PARAM_V_xfer];
    const double Vmax_up = d_parameters[PARAM_Vmax_up];
    const double k1_prime = d_parameters[PARAM_k1_prime];
    const double k2_prime = d_parameters[PARAM_k2_prime];
    const double k3 = d_parameters[PARAM_k3];
    const double k4 = d_parameters[PARAM_k4];
    const double max_sr = d_parameters[PARAM_max_sr];
    const double min_sr = d_parameters[PARAM_min_sr];
    const double Na_o = d_parameters[PARAM_Na_o];
    const double Cm = d_parameters[PARAM_Cm];
    const double F = d_parameters[PARAM_F];
    const double R = d_parameters[PARAM_R];
    const double T = d_parameters[PARAM_T];
    const double V_c = d_parameters[PARAM_V_c];
    const double is_stimulated = d_parameters[PARAM_is_stimulated];
    const double stim_amplitude = d_parameters[PARAM_stim_amplitude];
    const double K_o = d_parameters[PARAM_K_o];

    // Expressions for the Reversal potentials component
    const double E_Na = R * T * log(Na_o / Na_i) / F;
    const double E_K = R * T * log(K_o / K_i) / F;
    const double E_Ks = R * T * log((K_o + Na_o * P_kna) / (P_kna * Na_i + K_i)) / F;
    const double E_Ca = 0.5 * R * T * log(Ca_o / Ca_i) / F;

    // Expressions for the Inward rectifier potassium current component
    const double alpha_K1 = 0.1 / (1. + 6.14421235332821e-6 * exp(0.06 * V - 0.06 * E_K));
    const double beta_K1 = (0.367879441171442 * exp(0.1 * V - 0.1 * E_K)
                            + 3.06060402008027 * exp(0.0002 * V - 0.0002 * E_K))
                           / (1. + exp(0.5 * E_K - 0.5 * V));
    const double xK1_inf = alpha_K1 / (alpha_K1 + beta_K1);
    const double i_K1 = 0.430331482911935 * g_K1 * sqrt(K_o) * (-E_K + V) * xK1_inf;

    // Expressions for the Rapid time dependent potassium current component
    const double i_Kr = 0.430331482911935 * g_Kr * sqrt(K_o) * (-E_K + V) * Xr1 * Xr2;

    // Expressions for the Xr1 gate component
    const double xr1_inf = 1.0 / (1. + exp(-26. / 7. - V / 7.));
    const double alpha_xr1 = 450. / (1. + exp(-9. / 2. - V / 10.));
    const double beta_xr1 = 6. / (1. + exp(60. / 23. + 2. * V / 23.));
    const double tau_xr1 = alpha_xr1 * beta_xr1;
    const double dXr1_dt = (-Xr1 + xr1_inf) / tau_xr1;
    const double dXr1_dt_linearized = -1. / tau_xr1;
    d_states[padded_num_cells * STATE_Xr1 + thread_ind] =
            (expm1(dt * dXr1_dt_linearized)) * dXr1_dt / dXr1_dt_linearized + Xr1;

    // Expressions for the Xr2 gate component
    const double xr2_inf = 1.0 / (1. + exp(11. / 3. + V / 24.));
    const double alpha_xr2 = 3. / (1. + exp(-3. - V / 20.));
    const double beta_xr2 = 1.12 / (1. + exp(-3. + V / 20.));
    const double tau_xr2 = alpha_xr2 * beta_xr2;
    const double dXr2_dt = (-Xr2 + xr2_inf) / tau_xr2;
    const double dXr2_dt_linearized = -1. / tau_xr2;
    d_states[padded_num_cells * STATE_Xr2 + thread_ind] =
            (expm1(dt * dXr2_dt_linearized)) * dXr2_dt / dXr2_dt_linearized + Xr2;

    // Expressions for the Slow time dependent potassium current component
    const double i_Ks = g_Ks * (Xs * Xs) * (-E_Ks + V);

    // Expressions for the Xs gate component
    const double xs_inf = 1.0 / (1. + exp(-5. / 14. - V / 14.));
    const double alpha_xs = 1400. / sqrt(1. + exp(5. / 6. - V / 6.));
    const double beta_xs = 1.0 / (1. + exp(-7. / 3. + V / 15.));
    const double tau_xs = 80. + alpha_xs * beta_xs;
    const double dXs_dt = (-Xs + xs_inf) / tau_xs;
    const double dXs_dt_linearized = -1. / tau_xs;
    d_states[padded_num_cells * STATE_Xs + thread_ind] =
            (expm1(dt * dXs_dt_linearized)) * dXs_dt / dXs_dt_linearized + Xs;

    // Expressions for the Fast sodium current component
    const double i_Na = g_Na * (m * m * m) * (-E_Na + V) * h * j;

    // Expressions for the m gate component
    const double m_inf = 1.0
                         / ((1. + exp(-5686. / 903. - 100. * V / 903.))
                            * (1. + exp(-5686. / 903. - 100. * V / 903.)));
    const double alpha_m = 1.0 / (1. + exp(-12. - V / 5.));
    const double beta_m = 0.1 / (1. + exp(7. + V / 5.)) + 0.1 / (1. + exp(-1. / 4. + V / 200.));
    const double tau_m = alpha_m * beta_m;
    const double dm_dt = (-m + m_inf) / tau_m;
    const double dm_dt_linearized = -1. / tau_m;
    d_states[padded_num_cells * STATE_m + thread_ind] =
            (expm1(dt * dm_dt_linearized)) * dm_dt / dm_dt_linearized + m;

    // Expressions for the h gate component
    const double h_inf = 1.0
                         / ((1. + exp(7155. / 743. + 100. * V / 743.))
                            * (1. + exp(7155. / 743. + 100. * V / 743.)));
    const double alpha_h = (V < -40. ? 4.43126792958051e-7 * exp(-0.147058823529412 * V) : 0.);
    const double beta_h =
            (V < -40. ? 310000. * exp(0.3485 * V) + 2.7 * exp(0.079 * V)
                      : 0.77 / (0.13 + 0.0497581410839387 * exp(-0.0900900900900901 * V)));
    const double tau_h = 1.0 / (alpha_h + beta_h);
    const double dh_dt = (-h + h_inf) / tau_h;
    const double dh_dt_linearized = -1. / tau_h;
    d_states[padded_num_cells * STATE_h + thread_ind] =
            (expm1(dt * dh_dt_linearized)) * dh_dt / dh_dt_linearized + h;

    // Expressions for the j gate component
    const double j_inf = 1.0
                         / ((1. + exp(7155. / 743. + 100. * V / 743.))
                            * (1. + exp(7155. / 743. + 100. * V / 743.)));
    const double alpha_j =
            (V < -40. ? (37.78 + V) * (-25428. * exp(0.2444 * V) - 6.948e-6 * exp(-0.04391 * V))
                                / (1. + 50262745825.954 * exp(0.311 * V))
                      : 0.);
    const double beta_j =
            (V < -40. ? 0.02424 * exp(-0.01052 * V) / (1. + 0.00396086833990426 * exp(-0.1378 * V))
                      : 0.6 * exp(0.057 * V) / (1. + exp(-16. / 5. - V / 10.)));
    const double tau_j = 1.0 / (alpha_j + beta_j);
    const double dj_dt = (-j + j_inf) / tau_j;
    const double dj_dt_linearized = -1. / tau_j;
    d_states[padded_num_cells * STATE_j + thread_ind] =
            (expm1(dt * dj_dt_linearized)) * dj_dt / dj_dt_linearized + j;

    // Expressions for the Sodium background current component
    const double i_b_Na = g_bna * (-E_Na + V);

    // Expressions for the L_type Ca current component
    const double V_eff = -15. + V;
    const double i_CaL_factors = 4. * F * g_CaL
                                 * (-Ca_o + Ca_ss * exp(2. * F * V_eff / (R * T)) / 4.) * d * f * f2
                                 * fCass;
    const double i_CaL_fraction =
            (fabs(V_eff) < i_CaL_lim_delta
                     ? 0.5
                     : F * V_eff / (R * T * (expm1(2. * F * V_eff / (R * T)))));
    const double i_CaL = i_CaL_factors * i_CaL_fraction;

    // Expressions for the d gate component
    const double d_inf = 1.0 / (1. + exp(-16. / 15. - 2. * V / 15.));
    const double alpha_d = 0.25 + 1.4 / (1. + exp(-35. / 13. - V / 13.));
    const double beta_d = 1.4 / (1. + exp(1. + V / 5.));
    const double gamma_d = 1.0 / (1. + exp(5. / 2. - V / 20.));
    const double tau_d = alpha_d * beta_d + gamma_d;
    const double dd_dt = (-d + d_inf) / tau_d;
    const double dd_dt_linearized = -1. / tau_d;
    d_states[padded_num_cells * STATE_d + thread_ind] =
            (expm1(dt * dd_dt_linearized)) * dd_dt / dd_dt_linearized + d;

    // Expressions for the f gate component
    const double f_inf = 1.0 / (1. + exp(20. / 7. + V / 7.));
    const double tau_f = 20. + 180. / (1. + exp(3. + V / 10.))
                         + 200. / (1. + exp(13. / 10. - V / 10.))
                         + 1102.5 * exp(-((27. + V) * (27. + V)) / 225.);
    const double df_dt = (-f + f_inf) / tau_f;
    const double df_dt_linearized = -1. / tau_f;
    d_states[padded_num_cells * STATE_f + thread_ind] =
            (expm1(dt * df_dt_linearized)) * df_dt / df_dt_linearized + f;

    // Expressions for the F2 gate component
    const double f2_inf = 0.33 + 0.67 / (1. + exp(5. + V / 7.));
    const double tau_f2 = 31. / (1. + exp(5. / 2. - V / 10.)) + 80. / (1. + exp(3. + V / 10.))
                          + 562. * exp(-((27. + V) * (27. + V)) / 240.);
    const double df2_dt = (-f2 + f2_inf) / tau_f2;
    const double df2_dt_linearized = -1. / tau_f2;
    d_states[padded_num_cells * STATE_f2 + thread_ind] =
            (expm1(dt * df2_dt_linearized)) * df2_dt / df2_dt_linearized + f2;

    // Expressions for the FCass gate component
    const double fCass_inf = 0.4 + 0.6 / (1. + 400. * (Ca_ss * Ca_ss));
    const double tau_fCass = 2. + 80. / (1. + 400. * (Ca_ss * Ca_ss));
    const double dfCass_dt = (-fCass + fCass_inf) / tau_fCass;
    const double dfCass_dt_linearized = -1. / tau_fCass;
    d_states[padded_num_cells * STATE_fCass + thread_ind] =
            (expm1(dt * dfCass_dt_linearized)) * dfCass_dt / dfCass_dt_linearized + fCass;

    // Expressions for the Calcium background current component
    const double i_b_Ca = g_bca * (-E_Ca + V);

    // Expressions for the Transient outward current component
    const double i_to = g_to * (-E_K + V) * r * s;

    // Expressions for the s gate component
    const double s_inf =
            (celltype == 2. ? 1.0 / (1. + exp(28. / 5. + V / 5.)) : 1.0 / (1. + exp(4. + V / 5.)));
    const double tau_s = (celltype == 2. ? 8. + 1000. * exp(-((67. + V) * (67. + V)) / 1000.)
                                         : 3. + 5. / (1. + exp(-4. + V / 5.))
                                                   + 85. * exp(-((45. + V) * (45. + V)) / 320.));
    const double ds_dt = (-s + s_inf) / tau_s;
    const double ds_dt_linearized = -1. / tau_s;
    d_states[padded_num_cells * STATE_s + thread_ind] =
            (expm1(dt * ds_dt_linearized)) * ds_dt / ds_dt_linearized + s;

    // Expressions for the r gate component
    const double r_inf = 1.0 / (1. + exp(10. / 3. - V / 6.));
    const double tau_r = 0.8 + 9.5 * exp(-((40. + V) * (40. + V)) / 1800.);
    const double dr_dt = (-r + r_inf) / tau_r;
    const double dr_dt_linearized = -1. / tau_r;
    d_states[padded_num_cells * STATE_r + thread_ind] =
            (expm1(dt * dr_dt_linearized)) * dr_dt / dr_dt_linearized + r;

    // Expressions for the Sodium potassium pump current component
    const double i_NaK =
            K_o * P_NaK * Na_i
            / ((K_mNa + Na_i) * (K_mk + K_o)
               * (1. + 0.0353 * exp(-F * V / (R * T)) + 0.1245 * exp(-0.1 * F * V / (R * T))));

    // Expressions for the Sodium calcium exchanger current component
    const double i_NaCa =
            K_NaCa
            * (Ca_o * (Na_i * Na_i * Na_i) * exp(F * gamma * V / (R * T))
               - alpha * (Na_o * Na_o * Na_o) * Ca_i * exp(F * (-1. + gamma) * V / (R * T)))
            / ((1. + K_sat * exp(F * (-1. + gamma) * V / (R * T))) * (Ca_o + Km_Ca)
               * ((Km_Nai * Km_Nai * Km_Nai) + (Na_o * Na_o * Na_o)));

    // Expressions for the Calcium pump current component
    const double i_p_Ca = g_pCa * Ca_i / (K_pCa + Ca_i);

    // Expressions for the Potassium pump current component
    const double i_p_K = g_pK * (-E_K + V) / (1. + exp(1250. / 299. - 50. * V / 299.));

    // Expressions for the Calcium dynamics component
    const double i_up = Vmax_up / (1. + (K_up * K_up) / (Ca_i * Ca_i));
    const double i_leak = V_leak * (-Ca_i + Ca_SR);
    const double i_xfer = V_xfer * (-Ca_i + Ca_ss);
    const double kcasr = max_sr - (max_sr - min_sr) / (1. + (EC * EC) / (Ca_SR * Ca_SR));
    const double Ca_i_bufc = 1.0 / (1. + Buf_c * K_buf_c / ((K_buf_c + Ca_i) * (K_buf_c + Ca_i)));
    const double Ca_sr_bufsr =
            1.0 / (1. + Buf_sr * K_buf_sr / ((K_buf_sr + Ca_SR) * (K_buf_sr + Ca_SR)));
    const double Ca_ss_bufss =
            1.0 / (1. + Buf_ss * K_buf_ss / ((K_buf_ss + Ca_ss) * (K_buf_ss + Ca_ss)));
    const double dCa_i_dt = (V_sr * (-i_up + i_leak) / V_c
                             - Cm * (-2. * i_NaCa + i_b_Ca + i_p_Ca) / (2. * F * V_c) + i_xfer)
                            * Ca_i_bufc;
    d_states[padded_num_cells * STATE_Ca_i + thread_ind] = dt * dCa_i_dt + Ca_i;
    const double k1 = k1_prime / kcasr;
    const double k2 = k2_prime * kcasr;
    const double O = (Ca_ss * Ca_ss) * R_prime * k1 / (k3 + (Ca_ss * Ca_ss) * k1);
    const double dR_prime_dt = k4 * (1. - R_prime) - Ca_ss * R_prime * k2;
    d_states[padded_num_cells * STATE_R_prime + thread_ind] = dt * dR_prime_dt + R_prime;
    const double i_rel = V_rel * (-Ca_ss + Ca_SR) * O;
    const double dCa_SR_dt = (-i_leak - i_rel + i_up) * Ca_sr_bufsr;
    d_states[padded_num_cells * STATE_Ca_SR + thread_ind] = dt * dCa_SR_dt + Ca_SR;
    const double dCa_ss_dt =
            (V_sr * i_rel / V_ss - V_c * i_xfer / V_ss - Cm * i_CaL / (2. * F * V_ss))
            * Ca_ss_bufss;
    d_states[padded_num_cells * STATE_Ca_ss + thread_ind] = dt * dCa_ss_dt + Ca_ss;

    // Expressions for the Sodium dynamics component
    const double dNa_i_dt = Cm * (-i_Na - i_b_Na - 3. * i_NaCa - 3. * i_NaK) / (F * V_c);
    d_states[padded_num_cells * STATE_Na_i + thread_ind] = dt * dNa_i_dt + Na_i;

    // Expressions for the Membrane component
    const double i_Stim = (is_stimulated ? -stim_amplitude : 0.);
    const double dV_dt = -i_CaL - i_K1 - i_Kr - i_Ks - i_Na - i_NaCa - i_NaK - i_Stim - i_b_Ca
                         - i_b_Na - i_p_Ca - i_p_K - i_to;
    d_states[padded_num_cells * STATE_V + thread_ind] = dt * dV_dt + V;

    // Expressions for the Potassium dynamics component
    const double dK_i_dt =
            Cm * (-i_K1 - i_Kr - i_Ks - i_Stim - i_p_K - i_to + 2. * i_NaK) / (F * V_c);
    d_states[padded_num_cells * STATE_K_i + thread_ind] = dt * dK_i_dt + K_i;
}

// Compute a forward step using the generalised Rush-Larsen (GRL1) scheme to the
// TP06 ODE
__global__ void step_GRL1(double *__restrict d_states, const double t, const double dt,
                          const double *__restrict d_parameters, const unsigned int num_cells,
                          const unsigned int padded_num_cells)
{
    const int thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= num_cells)
        return; // number of nodes exceeded;

    // Assign states
    const double Xr1 = d_states[padded_num_cells * STATE_Xr1 + thread_ind];
    const double Xr2 = d_states[padded_num_cells * STATE_Xr2 + thread_ind];
    const double Xs = d_states[padded_num_cells * STATE_Xs + thread_ind];
    const double m = d_states[padded_num_cells * STATE_m + thread_ind];
    const double h = d_states[padded_num_cells * STATE_h + thread_ind];
    const double j = d_states[padded_num_cells * STATE_j + thread_ind];
    const double d = d_states[padded_num_cells * STATE_d + thread_ind];
    const double f = d_states[padded_num_cells * STATE_f + thread_ind];
    const double f2 = d_states[padded_num_cells * STATE_f2 + thread_ind];
    const double fCass = d_states[padded_num_cells * STATE_fCass + thread_ind];
    const double s = d_states[padded_num_cells * STATE_s + thread_ind];
    const double r = d_states[padded_num_cells * STATE_r + thread_ind];
    const double Ca_i = d_states[padded_num_cells * STATE_Ca_i + thread_ind];
    const double R_prime = d_states[padded_num_cells * STATE_R_prime + thread_ind];
    const double Ca_SR = d_states[padded_num_cells * STATE_Ca_SR + thread_ind];
    const double Ca_ss = d_states[padded_num_cells * STATE_Ca_ss + thread_ind];
    const double Na_i = d_states[padded_num_cells * STATE_Na_i + thread_ind];
    const double V = d_states[padded_num_cells * STATE_V + thread_ind];
    const double K_i = d_states[padded_num_cells * STATE_K_i + thread_ind];

    // Assign parameters
    const double celltype = d_parameters[PARAM_celltype];
    const double P_kna = d_parameters[PARAM_P_kna];
    const double g_K1 = d_parameters[PARAM_g_K1];
    const double g_Kr = d_parameters[PARAM_g_Kr];
    const double g_Ks = d_parameters[PARAM_g_Ks];
    const double g_Na = d_parameters[PARAM_g_Na];
    const double g_bna = d_parameters[PARAM_g_bna];
    const double g_CaL = d_parameters[PARAM_g_CaL];
    const double i_CaL_lim_delta = d_parameters[PARAM_i_CaL_lim_delta];
    const double g_bca = d_parameters[PARAM_g_bca];
    const double g_to = d_parameters[PARAM_g_to];
    const double K_mNa = d_parameters[PARAM_K_mNa];
    const double K_mk = d_parameters[PARAM_K_mk];
    const double P_NaK = d_parameters[PARAM_P_NaK];
    const double K_NaCa = d_parameters[PARAM_K_NaCa];
    const double K_sat = d_parameters[PARAM_K_sat];
    const double Km_Ca = d_parameters[PARAM_Km_Ca];
    const double Km_Nai = d_parameters[PARAM_Km_Nai];
    const double alpha = d_parameters[PARAM_alpha];
    const double gamma = d_parameters[PARAM_gamma];
    const double K_pCa = d_parameters[PARAM_K_pCa];
    const double g_pCa = d_parameters[PARAM_g_pCa];
    const double g_pK = d_parameters[PARAM_g_pK];
    const double Buf_c = d_parameters[PARAM_Buf_c];
    const double Buf_sr = d_parameters[PARAM_Buf_sr];
    const double Buf_ss = d_parameters[PARAM_Buf_ss];
    const double Ca_o = d_parameters[PARAM_Ca_o];
    const double EC = d_parameters[PARAM_EC];
    const double K_buf_c = d_parameters[PARAM_K_buf_c];
    const double K_buf_sr = d_parameters[PARAM_K_buf_sr];
    const double K_buf_ss = d_parameters[PARAM_K_buf_ss];
    const double K_up = d_parameters[PARAM_K_up];
    const double V_leak = d_parameters[PARAM_V_leak];
    const double V_rel = d_parameters[PARAM_V_rel];
    const double V_sr = d_parameters[PARAM_V_sr];
    const double V_ss = d_parameters[PARAM_V_ss];
    const double V_xfer = d_parameters[PARAM_V_xfer];
    const double Vmax_up = d_parameters[PARAM_Vmax_up];
    const double k1_prime = d_parameters[PARAM_k1_prime];
    const double k2_prime = d_parameters[PARAM_k2_prime];
    const double k3 = d_parameters[PARAM_k3];
    const double k4 = d_parameters[PARAM_k4];
    const double max_sr = d_parameters[PARAM_max_sr];
    const double min_sr = d_parameters[PARAM_min_sr];
    const double Na_o = d_parameters[PARAM_Na_o];
    const double Cm = d_parameters[PARAM_Cm];
    const double F = d_parameters[PARAM_F];
    const double R = d_parameters[PARAM_R];
    const double T = d_parameters[PARAM_T];
    const double V_c = d_parameters[PARAM_V_c];
    const double is_stimulated = d_parameters[PARAM_is_stimulated];
    const double stim_amplitude = d_parameters[PARAM_stim_amplitude];
    const double K_o = d_parameters[PARAM_K_o];

    // Expressions for the Reversal potentials component
    const double E_Na = R * T * log(Na_o / Na_i) / F;
    const double E_K = R * T * log(K_o / K_i) / F;
    const double E_Ks = R * T * log((K_o + Na_o * P_kna) / (P_kna * Na_i + K_i)) / F;
    const double E_Ca = 0.5 * R * T * log(Ca_o / Ca_i) / F;

    // Expressions for the Inward rectifier potassium current component
    const double alpha_K1 = 0.1 / (1. + 6.14421235332821e-6 * exp(0.06 * V - 0.06 * E_K));
    const double beta_K1 = (0.367879441171442 * exp(0.1 * V - 0.1 * E_K)
                            + 3.06060402008027 * exp(0.0002 * V - 0.0002 * E_K))
                           / (1. + exp(0.5 * E_K - 0.5 * V));
    const double xK1_inf = alpha_K1 / (alpha_K1 + beta_K1);
    const double i_K1 = 0.430331482911935 * g_K1 * sqrt(K_o) * (-E_K + V) * xK1_inf;

    // Expressions for the Rapid time dependent potassium current component
    const double i_Kr = 0.430331482911935 * g_Kr * sqrt(K_o) * (-E_K + V) * Xr1 * Xr2;

    // Expressions for the Xr1 gate component
    const double xr1_inf = 1.0 / (1. + exp(-26. / 7. - V / 7.));
    const double alpha_xr1 = 450. / (1. + exp(-9. / 2. - V / 10.));
    const double beta_xr1 = 6. / (1. + exp(60. / 23. + 2. * V / 23.));
    const double tau_xr1 = alpha_xr1 * beta_xr1;
    const double dXr1_dt = (-Xr1 + xr1_inf) / tau_xr1;
    const double dXr1_dt_linearized = -1. / tau_xr1;
    d_states[padded_num_cells * STATE_Xr1 + thread_ind] =
            (expm1(dt * dXr1_dt_linearized)) * dXr1_dt / dXr1_dt_linearized + Xr1;

    // Expressions for the Xr2 gate component
    const double xr2_inf = 1.0 / (1. + exp(11. / 3. + V / 24.));
    const double alpha_xr2 = 3. / (1. + exp(-3. - V / 20.));
    const double beta_xr2 = 1.12 / (1. + exp(-3. + V / 20.));
    const double tau_xr2 = alpha_xr2 * beta_xr2;
    const double dXr2_dt = (-Xr2 + xr2_inf) / tau_xr2;
    const double dXr2_dt_linearized = -1. / tau_xr2;
    d_states[padded_num_cells * STATE_Xr2 + thread_ind] =
            (expm1(dt * dXr2_dt_linearized)) * dXr2_dt / dXr2_dt_linearized + Xr2;

    // Expressions for the Slow time dependent potassium current component
    const double i_Ks = g_Ks * (Xs * Xs) * (-E_Ks + V);

    // Expressions for the Xs gate component
    const double xs_inf = 1.0 / (1. + exp(-5. / 14. - V / 14.));
    const double alpha_xs = 1400. / sqrt(1. + exp(5. / 6. - V / 6.));
    const double beta_xs = 1.0 / (1. + exp(-7. / 3. + V / 15.));
    const double tau_xs = 80. + alpha_xs * beta_xs;
    const double dXs_dt = (-Xs + xs_inf) / tau_xs;
    const double dXs_dt_linearized = -1. / tau_xs;
    d_states[padded_num_cells * STATE_Xs + thread_ind] =
            (expm1(dt * dXs_dt_linearized)) * dXs_dt / dXs_dt_linearized + Xs;

    // Expressions for the Fast sodium current component
    const double i_Na = g_Na * (m * m * m) * (-E_Na + V) * h * j;

    // Expressions for the m gate component
    const double m_inf = 1.0
                         / ((1. + exp(-5686. / 903. - 100. * V / 903.))
                            * (1. + exp(-5686. / 903. - 100. * V / 903.)));
    const double alpha_m = 1.0 / (1. + exp(-12. - V / 5.));
    const double beta_m = 0.1 / (1. + exp(7. + V / 5.)) + 0.1 / (1. + exp(-1. / 4. + V / 200.));
    const double tau_m = alpha_m * beta_m;
    const double dm_dt = (-m + m_inf) / tau_m;
    const double dm_dt_linearized = -1. / tau_m;
    d_states[padded_num_cells * STATE_m + thread_ind] =
            (expm1(dt * dm_dt_linearized)) * dm_dt / dm_dt_linearized + m;

    // Expressions for the h gate component
    const double h_inf = 1.0
                         / ((1. + exp(7155. / 743. + 100. * V / 743.))
                            * (1. + exp(7155. / 743. + 100. * V / 743.)));
    const double alpha_h = (V < -40. ? 4.43126792958051e-7 * exp(-0.147058823529412 * V) : 0.);
    const double beta_h =
            (V < -40. ? 310000. * exp(0.3485 * V) + 2.7 * exp(0.079 * V)
                      : 0.77 / (0.13 + 0.0497581410839387 * exp(-0.0900900900900901 * V)));
    const double tau_h = 1.0 / (alpha_h + beta_h);
    const double dh_dt = (-h + h_inf) / tau_h;
    const double dh_dt_linearized = -1. / tau_h;
    d_states[padded_num_cells * STATE_h + thread_ind] =
            (expm1(dt * dh_dt_linearized)) * dh_dt / dh_dt_linearized + h;

    // Expressions for the j gate component
    const double j_inf = 1.0
                         / ((1. + exp(7155. / 743. + 100. * V / 743.))
                            * (1. + exp(7155. / 743. + 100. * V / 743.)));
    const double alpha_j =
            (V < -40. ? (37.78 + V) * (-25428. * exp(0.2444 * V) - 6.948e-6 * exp(-0.04391 * V))
                                / (1. + 50262745825.954 * exp(0.311 * V))
                      : 0.);
    const double beta_j =
            (V < -40. ? 0.02424 * exp(-0.01052 * V) / (1. + 0.00396086833990426 * exp(-0.1378 * V))
                      : 0.6 * exp(0.057 * V) / (1. + exp(-16. / 5. - V / 10.)));
    const double tau_j = 1.0 / (alpha_j + beta_j);
    const double dj_dt = (-j + j_inf) / tau_j;
    const double dj_dt_linearized = -1. / tau_j;
    d_states[padded_num_cells * STATE_j + thread_ind] =
            (expm1(dt * dj_dt_linearized)) * dj_dt / dj_dt_linearized + j;

    // Expressions for the Sodium background current component
    const double i_b_Na = g_bna * (-E_Na + V);

    // Expressions for the L_type Ca current component
    const double V_eff = -15. + V;
    const double i_CaL_factors = 4. * F * g_CaL
                                 * (-Ca_o + Ca_ss * exp(2. * F * V_eff / (R * T)) / 4.) * d * f * f2
                                 * fCass;
    const double i_CaL_fraction =
            (fabs(V_eff) < i_CaL_lim_delta
                     ? 0.5
                     : F * V_eff / (R * T * (expm1(2. * F * V_eff / (R * T)))));
    const double i_CaL = i_CaL_factors * i_CaL_fraction;

    // Expressions for the d gate component
    const double d_inf = 1.0 / (1. + exp(-16. / 15. - 2. * V / 15.));
    const double alpha_d = 0.25 + 1.4 / (1. + exp(-35. / 13. - V / 13.));
    const double beta_d = 1.4 / (1. + exp(1. + V / 5.));
    const double gamma_d = 1.0 / (1. + exp(5. / 2. - V / 20.));
    const double tau_d = alpha_d * beta_d + gamma_d;
    const double dd_dt = (-d + d_inf) / tau_d;
    const double dd_dt_linearized = -1. / tau_d;
    d_states[padded_num_cells * STATE_d + thread_ind] =
            (expm1(dt * dd_dt_linearized)) * dd_dt / dd_dt_linearized + d;

    // Expressions for the f gate component
    const double f_inf = 1.0 / (1. + exp(20. / 7. + V / 7.));
    const double tau_f = 20. + 180. / (1. + exp(3. + V / 10.))
                         + 200. / (1. + exp(13. / 10. - V / 10.))
                         + 1102.5 * exp(-((27. + V) * (27. + V)) / 225.);
    const double df_dt = (-f + f_inf) / tau_f;
    const double df_dt_linearized = -1. / tau_f;
    d_states[padded_num_cells * STATE_f + thread_ind] =
            (expm1(dt * df_dt_linearized)) * df_dt / df_dt_linearized + f;

    // Expressions for the F2 gate component
    const double f2_inf = 0.33 + 0.67 / (1. + exp(5. + V / 7.));
    const double tau_f2 = 31. / (1. + exp(5. / 2. - V / 10.)) + 80. / (1. + exp(3. + V / 10.))
                          + 562. * exp(-((27. + V) * (27. + V)) / 240.);
    const double df2_dt = (-f2 + f2_inf) / tau_f2;
    const double df2_dt_linearized = -1. / tau_f2;
    d_states[padded_num_cells * STATE_f2 + thread_ind] =
            (expm1(dt * df2_dt_linearized)) * df2_dt / df2_dt_linearized + f2;

    // Expressions for the FCass gate component
    const double fCass_inf = 0.4 + 0.6 / (1. + 400. * (Ca_ss * Ca_ss));
    const double tau_fCass = 2. + 80. / (1. + 400. * (Ca_ss * Ca_ss));
    const double dfCass_dt = (-fCass + fCass_inf) / tau_fCass;
    const double dfCass_dt_linearized = -1. / tau_fCass;
    d_states[padded_num_cells * STATE_fCass + thread_ind] =
            (expm1(dt * dfCass_dt_linearized)) * dfCass_dt / dfCass_dt_linearized + fCass;

    // Expressions for the Calcium background current component
    const double i_b_Ca = g_bca * (-E_Ca + V);

    // Expressions for the Transient outward current component
    const double i_to = g_to * (-E_K + V) * r * s;

    // Expressions for the s gate component
    const double s_inf =
            (celltype == 2. ? 1.0 / (1. + exp(28. / 5. + V / 5.)) : 1.0 / (1. + exp(4. + V / 5.)));
    const double tau_s = (celltype == 2. ? 8. + 1000. * exp(-((67. + V) * (67. + V)) / 1000.)
                                         : 3. + 5. / (1. + exp(-4. + V / 5.))
                                                   + 85. * exp(-((45. + V) * (45. + V)) / 320.));
    const double ds_dt = (-s + s_inf) / tau_s;
    const double ds_dt_linearized = -1. / tau_s;
    d_states[padded_num_cells * STATE_s + thread_ind] =
            (expm1(dt * ds_dt_linearized)) * ds_dt / ds_dt_linearized + s;

    // Expressions for the r gate component
    const double r_inf = 1.0 / (1. + exp(10. / 3. - V / 6.));
    const double tau_r = 0.8 + 9.5 * exp(-((40. + V) * (40. + V)) / 1800.);
    const double dr_dt = (-r + r_inf) / tau_r;
    const double dr_dt_linearized = -1. / tau_r;
    d_states[padded_num_cells * STATE_r + thread_ind] =
            (expm1(dt * dr_dt_linearized)) * dr_dt / dr_dt_linearized + r;

    // Expressions for the Sodium potassium pump current component
    const double i_NaK =
            K_o * P_NaK * Na_i
            / ((K_mNa + Na_i) * (K_mk + K_o)
               * (1. + 0.0353 * exp(-F * V / (R * T)) + 0.1245 * exp(-0.1 * F * V / (R * T))));

    // Expressions for the Sodium calcium exchanger current component
    const double i_NaCa =
            K_NaCa
            * (Ca_o * (Na_i * Na_i * Na_i) * exp(F * gamma * V / (R * T))
               - alpha * (Na_o * Na_o * Na_o) * Ca_i * exp(F * (-1. + gamma) * V / (R * T)))
            / ((1. + K_sat * exp(F * (-1. + gamma) * V / (R * T))) * (Ca_o + Km_Ca)
               * ((Km_Nai * Km_Nai * Km_Nai) + (Na_o * Na_o * Na_o)));

    // Expressions for the Calcium pump current component
    const double i_p_Ca = g_pCa * Ca_i / (K_pCa + Ca_i);

    // Expressions for the Potassium pump current component
    const double i_p_K = g_pK * (-E_K + V) / (1. + exp(1250. / 299. - 50. * V / 299.));

    // Expressions for the Calcium dynamics component
    const double i_up = Vmax_up / (1. + (K_up * K_up) / (Ca_i * Ca_i));
    const double i_leak = V_leak * (-Ca_i + Ca_SR);
    const double i_xfer = V_xfer * (-Ca_i + Ca_ss);
    const double kcasr = max_sr - (max_sr - min_sr) / (1. + (EC * EC) / (Ca_SR * Ca_SR));
    const double Ca_i_bufc = 1.0 / (1. + Buf_c * K_buf_c / ((K_buf_c + Ca_i) * (K_buf_c + Ca_i)));
    const double Ca_sr_bufsr =
            1.0 / (1. + Buf_sr * K_buf_sr / ((K_buf_sr + Ca_SR) * (K_buf_sr + Ca_SR)));
    const double Ca_ss_bufss =
            1.0 / (1. + Buf_ss * K_buf_ss / ((K_buf_ss + Ca_ss) * (K_buf_ss + Ca_ss)));
    const double dCa_i_dt = (V_sr * (-i_up + i_leak) / V_c
                             - Cm * (-2. * i_NaCa + i_b_Ca + i_p_Ca) / (2. * F * V_c) + i_xfer)
                            * Ca_i_bufc;
    const double dCa_i_bufc_dCa_i =
            2. * Buf_c * K_buf_c
            / (((1. + Buf_c * K_buf_c / ((K_buf_c + Ca_i) * (K_buf_c + Ca_i)))
                * (1. + Buf_c * K_buf_c / ((K_buf_c + Ca_i) * (K_buf_c + Ca_i))))
               * ((K_buf_c + Ca_i) * (K_buf_c + Ca_i) * (K_buf_c + Ca_i)));
    const double dE_Ca_dCa_i = -0.5 * R * T / (F * Ca_i);
    const double di_NaCa_dCa_i =
            -K_NaCa * alpha * (Na_o * Na_o * Na_o) * exp(F * (-1. + gamma) * V / (R * T))
            / ((1. + K_sat * exp(F * (-1. + gamma) * V / (R * T))) * (Ca_o + Km_Ca)
               * ((Km_Nai * Km_Nai * Km_Nai) + (Na_o * Na_o * Na_o)));
    const double di_p_Ca_dCa_i =
            g_pCa / (K_pCa + Ca_i) - g_pCa * Ca_i / ((K_pCa + Ca_i) * (K_pCa + Ca_i));
    const double di_up_dCa_i =
            2. * Vmax_up * (K_up * K_up)
            / (((1. + (K_up * K_up) / (Ca_i * Ca_i)) * (1. + (K_up * K_up) / (Ca_i * Ca_i)))
               * (Ca_i * Ca_i * Ca_i));
    const double dCa_i_dt_linearized =
            (-V_xfer + V_sr * (-V_leak - di_up_dCa_i) / V_c
             - Cm * (-2. * di_NaCa_dCa_i - g_bca * dE_Ca_dCa_i + di_p_Ca_dCa_i) / (2. * F * V_c))
                    * Ca_i_bufc
            + (V_sr * (-i_up + i_leak) / V_c
               - Cm * (-2. * i_NaCa + i_b_Ca + i_p_Ca) / (2. * F * V_c) + i_xfer)
                      * dCa_i_bufc_dCa_i;
    d_states[padded_num_cells * STATE_Ca_i + thread_ind] =
            Ca_i
            + (fabs(dCa_i_dt_linearized) > 1.0e-8
                       ? (expm1(dt * dCa_i_dt_linearized)) * dCa_i_dt / dCa_i_dt_linearized
                       : dt * dCa_i_dt);
    const double k1 = k1_prime / kcasr;
    const double k2 = k2_prime * kcasr;
    const double O = (Ca_ss * Ca_ss) * R_prime * k1 / (k3 + (Ca_ss * Ca_ss) * k1);
    const double dR_prime_dt = k4 * (1. - R_prime) - Ca_ss * R_prime * k2;
    const double dR_prime_dt_linearized = -k4 - Ca_ss * k2;
    d_states[padded_num_cells * STATE_R_prime + thread_ind] =
            (fabs(dR_prime_dt_linearized) > 1.0e-8
                     ? (expm1(dt * dR_prime_dt_linearized)) * dR_prime_dt / dR_prime_dt_linearized
                     : dt * dR_prime_dt)
            + R_prime;
    const double i_rel = V_rel * (-Ca_ss + Ca_SR) * O;
    const double dCa_SR_dt = (-i_leak - i_rel + i_up) * Ca_sr_bufsr;
    const double dCa_sr_bufsr_dCa_SR =
            2. * Buf_sr * K_buf_sr
            / (((1. + Buf_sr * K_buf_sr / ((K_buf_sr + Ca_SR) * (K_buf_sr + Ca_SR)))
                * (1. + Buf_sr * K_buf_sr / ((K_buf_sr + Ca_SR) * (K_buf_sr + Ca_SR))))
               * ((K_buf_sr + Ca_SR) * (K_buf_sr + Ca_SR) * (K_buf_sr + Ca_SR)));
    const double dO_dk1 = (Ca_ss * Ca_ss) * R_prime / (k3 + (Ca_ss * Ca_ss) * k1)
                          - (((Ca_ss) * (Ca_ss)) * ((Ca_ss) * (Ca_ss))) * R_prime * k1
                                    / ((k3 + (Ca_ss * Ca_ss) * k1) * (k3 + (Ca_ss * Ca_ss) * k1));
    const double dk1_dkcasr = -k1_prime / (kcasr * kcasr);
    const double dkcasr_dCa_SR =
            -2. * (EC * EC) * (max_sr - min_sr)
            / (((1. + (EC * EC) / (Ca_SR * Ca_SR)) * (1. + (EC * EC) / (Ca_SR * Ca_SR)))
               * (Ca_SR * Ca_SR * Ca_SR));
    const double di_rel_dCa_SR =
            V_rel * O + V_rel * (-Ca_ss + Ca_SR) * dO_dk1 * dk1_dkcasr * dkcasr_dCa_SR;
    const double di_rel_dO = V_rel * (-Ca_ss + Ca_SR);
    const double dCa_SR_dt_linearized =
            (-V_leak - di_rel_dCa_SR - dO_dk1 * di_rel_dO * dk1_dkcasr * dkcasr_dCa_SR)
                    * Ca_sr_bufsr
            + (-i_leak - i_rel + i_up) * dCa_sr_bufsr_dCa_SR;
    d_states[padded_num_cells * STATE_Ca_SR + thread_ind] =
            Ca_SR
            + (fabs(dCa_SR_dt_linearized) > 1.0e-8
                       ? (expm1(dt * dCa_SR_dt_linearized)) * dCa_SR_dt / dCa_SR_dt_linearized
                       : dt * dCa_SR_dt);
    const double dCa_ss_dt =
            (V_sr * i_rel / V_ss - V_c * i_xfer / V_ss - Cm * i_CaL / (2. * F * V_ss))
            * Ca_ss_bufss;
    const double dCa_ss_bufss_dCa_ss =
            2. * Buf_ss * K_buf_ss
            / (((1. + Buf_ss * K_buf_ss / ((K_buf_ss + Ca_ss) * (K_buf_ss + Ca_ss)))
                * (1. + Buf_ss * K_buf_ss / ((K_buf_ss + Ca_ss) * (K_buf_ss + Ca_ss))))
               * ((K_buf_ss + Ca_ss) * (K_buf_ss + Ca_ss) * (K_buf_ss + Ca_ss)));
    const double dO_dCa_ss = -2. * (Ca_ss * Ca_ss * Ca_ss) * (k1 * k1) * R_prime
                                     / ((k3 + (Ca_ss * Ca_ss) * k1) * (k3 + (Ca_ss * Ca_ss) * k1))
                             + 2. * Ca_ss * R_prime * k1 / (k3 + (Ca_ss * Ca_ss) * k1);
    const double di_CaL_factors_dCa_ss =
            F * g_CaL * d * exp(2. * F * V_eff / (R * T)) * f * f2 * fCass;
    const double di_rel_dCa_ss = -V_rel * O + V_rel * (-Ca_ss + Ca_SR) * dO_dCa_ss;
    const double dCa_ss_dt_linearized =
            (V_sr * (dO_dCa_ss * di_rel_dO + di_rel_dCa_ss) / V_ss - V_c * V_xfer / V_ss
             - Cm * di_CaL_factors_dCa_ss * i_CaL_fraction / (2. * F * V_ss))
                    * Ca_ss_bufss
            + (V_sr * i_rel / V_ss - V_c * i_xfer / V_ss - Cm * i_CaL / (2. * F * V_ss))
                      * dCa_ss_bufss_dCa_ss;
    d_states[padded_num_cells * STATE_Ca_ss + thread_ind] =
            Ca_ss
            + (fabs(dCa_ss_dt_linearized) > 1.0e-8
                       ? (expm1(dt * dCa_ss_dt_linearized)) * dCa_ss_dt / dCa_ss_dt_linearized
                       : dt * dCa_ss_dt);

    // Expressions for the Sodium dynamics component
    const double dNa_i_dt = Cm * (-i_Na - i_b_Na - 3. * i_NaCa - 3. * i_NaK) / (F * V_c);
    const double dE_Na_dNa_i = -R * T / (F * Na_i);
    const double di_Na_dE_Na = -g_Na * (m * m * m) * h * j;
    const double di_NaCa_dNa_i =
            3. * Ca_o * K_NaCa * (Na_i * Na_i) * exp(F * gamma * V / (R * T))
            / ((1. + K_sat * exp(F * (-1. + gamma) * V / (R * T))) * (Ca_o + Km_Ca)
               * ((Km_Nai * Km_Nai * Km_Nai) + (Na_o * Na_o * Na_o)));
    const double di_NaK_dNa_i = K_o * P_NaK
                                        / ((K_mNa + Na_i) * (K_mk + K_o)
                                           * (1. + 0.0353 * exp(-F * V / (R * T))
                                              + 0.1245 * exp(-0.1 * F * V / (R * T))))
                                - K_o * P_NaK * Na_i
                                          / (((K_mNa + Na_i) * (K_mNa + Na_i)) * (K_mk + K_o)
                                             * (1. + 0.0353 * exp(-F * V / (R * T))
                                                + 0.1245 * exp(-0.1 * F * V / (R * T))));
    const double dNa_i_dt_linearized = Cm
                                       * (-3. * di_NaCa_dNa_i - 3. * di_NaK_dNa_i
                                          + g_bna * dE_Na_dNa_i - dE_Na_dNa_i * di_Na_dE_Na)
                                       / (F * V_c);
    d_states[padded_num_cells * STATE_Na_i + thread_ind] =
            Na_i
            + (fabs(dNa_i_dt_linearized) > 1.0e-8
                       ? (expm1(dt * dNa_i_dt_linearized)) * dNa_i_dt / dNa_i_dt_linearized
                       : dt * dNa_i_dt);

    // Expressions for the Membrane component
    const double i_Stim = (is_stimulated ? -stim_amplitude : 0.);
    const double dV_dt = -i_CaL - i_K1 - i_Kr - i_Ks - i_Na - i_NaCa - i_NaK - i_Stim - i_b_Ca
                         - i_b_Na - i_p_Ca - i_p_K - i_to;
    const double dalpha_K1_dV = -3.68652741199693e-8 * exp(0.06 * V - 0.06 * E_K)
                                / ((1. + 6.14421235332821e-6 * exp(0.06 * V - 0.06 * E_K))
                                   * (1. + 6.14421235332821e-6 * exp(0.06 * V - 0.06 * E_K)));
    const double dbeta_K1_dV =
            (0.000612120804016053 * exp(0.0002 * V - 0.0002 * E_K)
             + 0.0367879441171442 * exp(0.1 * V - 0.1 * E_K))
                    / (1. + exp(0.5 * E_K - 0.5 * V))
            + 0.5
                      * (0.367879441171442 * exp(0.1 * V - 0.1 * E_K)
                         + 3.06060402008027 * exp(0.0002 * V - 0.0002 * E_K))
                      * exp(0.5 * E_K - 0.5 * V)
                      / ((1. + exp(0.5 * E_K - 0.5 * V)) * (1. + exp(0.5 * E_K - 0.5 * V)));
    const double di_CaL_factors_dV_eff = 2. * g_CaL * (F * F) * Ca_ss * d
                                         * exp(2. * F * V_eff / (R * T)) * f * f2 * fCass / (R * T);
    const double di_CaL_fraction_dV_eff =
            (fabs(V_eff) < i_CaL_lim_delta
                     ? 0.
                     : F / (R * T * (expm1(2. * F * V_eff / (R * T))))
                               - 2. * (F * F) * V_eff * exp(2. * F * V_eff / (R * T))
                                         / ((R * R) * (T * T)
                                            * ((expm1(2. * F * V_eff / (R * T)))
                                               * (expm1(2. * F * V_eff / (R * T))))));
    const double dxK1_inf_dalpha_K1 =
            1.0 / (alpha_K1 + beta_K1) - alpha_K1 / ((alpha_K1 + beta_K1) * (alpha_K1 + beta_K1));
    const double dxK1_inf_dbeta_K1 = -alpha_K1 / ((alpha_K1 + beta_K1) * (alpha_K1 + beta_K1));
    const double di_K1_dV =
            0.430331482911935 * g_K1 * sqrt(K_o) * xK1_inf
            + 0.430331482911935 * g_K1 * sqrt(K_o) * (-E_K + V)
                      * (dalpha_K1_dV * dxK1_inf_dalpha_K1 + dbeta_K1_dV * dxK1_inf_dbeta_K1);
    const double di_K1_dxK1_inf = 0.430331482911935 * g_K1 * sqrt(K_o) * (-E_K + V);
    const double di_Kr_dV = 0.430331482911935 * g_Kr * sqrt(K_o) * Xr1 * Xr2;
    const double di_Ks_dV = g_Ks * (Xs * Xs);
    const double di_Na_dV = g_Na * (m * m * m) * h * j;
    const double di_NaCa_dV =
            K_NaCa
                    * (Ca_o * F * gamma * (Na_i * Na_i * Na_i) * exp(F * gamma * V / (R * T))
                               / (R * T)
                       - F * alpha * (Na_o * Na_o * Na_o) * (-1. + gamma) * Ca_i
                                 * exp(F * (-1. + gamma) * V / (R * T)) / (R * T))
                    / ((1. + K_sat * exp(F * (-1. + gamma) * V / (R * T))) * (Ca_o + Km_Ca)
                       * ((Km_Nai * Km_Nai * Km_Nai) + (Na_o * Na_o * Na_o)))
            - F * K_NaCa * K_sat * (-1. + gamma)
                      * (Ca_o * (Na_i * Na_i * Na_i) * exp(F * gamma * V / (R * T))
                         - alpha * (Na_o * Na_o * Na_o) * Ca_i
                                   * exp(F * (-1. + gamma) * V / (R * T)))
                      * exp(F * (-1. + gamma) * V / (R * T))
                      / (R * T
                         * ((1. + K_sat * exp(F * (-1. + gamma) * V / (R * T)))
                            * (1. + K_sat * exp(F * (-1. + gamma) * V / (R * T))))
                         * (Ca_o + Km_Ca) * ((Km_Nai * Km_Nai * Km_Nai) + (Na_o * Na_o * Na_o)));
    const double di_NaK_dV =
            K_o * P_NaK
            * (0.0353 * F * exp(-F * V / (R * T)) / (R * T)
               + 0.01245 * F * exp(-0.1 * F * V / (R * T)) / (R * T))
            * Na_i
            / ((K_mNa + Na_i) * (K_mk + K_o)
               * ((1. + 0.0353 * exp(-F * V / (R * T)) + 0.1245 * exp(-0.1 * F * V / (R * T)))
                  * (1. + 0.0353 * exp(-F * V / (R * T)) + 0.1245 * exp(-0.1 * F * V / (R * T)))));
    const double di_p_K_dV = g_pK / (1. + exp(1250. / 299. - 50. * V / 299.))
                             + 50. * g_pK * (-E_K + V) * exp(1250. / 299. - 50. * V / 299.)
                                       / (299.
                                          * ((1. + exp(1250. / 299. - 50. * V / 299.))
                                             * (1. + exp(1250. / 299. - 50. * V / 299.))));
    const double di_to_dV = g_to * r * s;
    const double dV_dt_linearized =
            -g_bca - g_bna - di_K1_dV - di_Kr_dV - di_Ks_dV - di_NaCa_dV - di_NaK_dV - di_Na_dV
            - di_p_K_dV - di_to_dV
            - (dalpha_K1_dV * dxK1_inf_dalpha_K1 + dbeta_K1_dV * dxK1_inf_dbeta_K1) * di_K1_dxK1_inf
            - di_CaL_factors_dV_eff * i_CaL_fraction - di_CaL_fraction_dV_eff * i_CaL_factors;
    d_states[padded_num_cells * STATE_V + thread_ind] =
            (fabs(dV_dt_linearized) > 1.0e-8
                     ? (expm1(dt * dV_dt_linearized)) * dV_dt / dV_dt_linearized
                     : dt * dV_dt)
            + V;

    // Expressions for the Potassium dynamics component
    const double dK_i_dt =
            Cm * (-i_K1 - i_Kr - i_Ks - i_Stim - i_p_K - i_to + 2. * i_NaK) / (F * V_c);
    const double dE_K_dK_i = -R * T / (F * K_i);
    const double dE_Ks_dK_i = -R * T / (F * (P_kna * Na_i + K_i));
    const double dalpha_K1_dE_K = 3.68652741199693e-8 * exp(0.06 * V - 0.06 * E_K)
                                  / ((1. + 6.14421235332821e-6 * exp(0.06 * V - 0.06 * E_K))
                                     * (1. + 6.14421235332821e-6 * exp(0.06 * V - 0.06 * E_K)));
    const double dbeta_K1_dE_K =
            (-0.000612120804016053 * exp(0.0002 * V - 0.0002 * E_K)
             - 0.0367879441171442 * exp(0.1 * V - 0.1 * E_K))
                    / (1. + exp(0.5 * E_K - 0.5 * V))
            - 0.5
                      * (0.367879441171442 * exp(0.1 * V - 0.1 * E_K)
                         + 3.06060402008027 * exp(0.0002 * V - 0.0002 * E_K))
                      * exp(0.5 * E_K - 0.5 * V)
                      / ((1. + exp(0.5 * E_K - 0.5 * V)) * (1. + exp(0.5 * E_K - 0.5 * V)));
    const double di_K1_dE_K =
            -0.430331482911935 * g_K1 * sqrt(K_o) * xK1_inf
            + 0.430331482911935 * g_K1 * sqrt(K_o) * (-E_K + V)
                      * (dalpha_K1_dE_K * dxK1_inf_dalpha_K1 + dbeta_K1_dE_K * dxK1_inf_dbeta_K1);
    const double di_Kr_dE_K = -0.430331482911935 * g_Kr * sqrt(K_o) * Xr1 * Xr2;
    const double di_Ks_dE_Ks = -g_Ks * (Xs * Xs);
    const double di_p_K_dE_K = -g_pK / (1. + exp(1250. / 299. - 50. * V / 299.));
    const double di_to_dE_K = -g_to * r * s;
    const double dK_i_dt_linearized =
            Cm
            * (-(dE_K_dK_i * dalpha_K1_dE_K * dxK1_inf_dalpha_K1
                 + dE_K_dK_i * dbeta_K1_dE_K * dxK1_inf_dbeta_K1)
                       * di_K1_dxK1_inf
               - dE_K_dK_i * di_K1_dE_K - dE_K_dK_i * di_Kr_dE_K - dE_K_dK_i * di_p_K_dE_K
               - dE_K_dK_i * di_to_dE_K - dE_Ks_dK_i * di_Ks_dE_Ks)
            / (F * V_c);
    d_states[padded_num_cells * STATE_K_i + thread_ind] =
            K_i
            + (fabs(dK_i_dt_linearized) > 1.0e-8
                       ? (expm1(dt * dK_i_dt_linearized)) * dK_i_dt / dK_i_dt_linearized
                       : dt * dK_i_dt);
}

size_t ceil_div(size_t a, size_t b)
{
    return (a + b - 1) / b;
}

size_t ceil_to_multiple(size_t a, size_t b)
{
    return ceil_div(a, b) * b;
}

int choose_threadblock_size(int device_number, enum enum_scheme kernel)
{
    cudaDeviceProp deviceProp;
    checkCuda(cudaGetDeviceProperties(&deviceProp, device_number));

    int threadblock_size = 0;
    int num_threads_max = 0;
    for (int tbs = 32; tbs <= deviceProp.maxThreadsPerBlock; tbs *= 2) {
        int num_blocks;
        switch (kernel) {
        case scheme_arg_FE:
            checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, step_FE, tbs, 0));
            break;

        case scheme_arg_RL:
            checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, step_RL, tbs, 0));
            break;

        case scheme_arg_GRL1:
            checkCuda(
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, step_GRL1, tbs, 0));
            break;
        default:
            printf("choose_threadblock_size(): Unsupported scheme/kernel: %d\n", kernel);
            exit(EXIT_FAILURE);
        }
        int num_threads = num_blocks * tbs;
        // printf("With tbs=%d, max threads is %d\n", tbs, num_threads);
        if (num_threads > num_threads_max) {
            num_threads_max = num_threads;
            threadblock_size = tbs;
        }
    }
    return threadblock_size;
}

void bench(int num_cells, int num_timesteps, double dt, enum enum_scheme scheme, int device_id)
{

    // store each state variable contiguously in memory
    // pad so that each state variable starts at a 128-byte boundary (cache line)
    int padded_num_cells = ceil_to_multiple(num_cells, 128 / sizeof(double));
    size_t states_size = NUM_STATES * (long) padded_num_cells * sizeof(double);
    size_t parameters_size = NUM_PARAMS * sizeof(double);

    double *h_states;
    double *h_parameters;
    double *d_states;
    double *d_parameters;

    cudaMallocHost(&h_states, states_size);
    cudaMallocHost(&h_parameters, parameters_size);

    cudaMalloc(&d_states, states_size);
    cudaMalloc(&d_parameters, parameters_size);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // choose threadblock size that maximises occupancy
    cudaSetDevice(device_id);
    int threadblock_size = choose_threadblock_size(device_id, scheme);
    int threadblocks = ceil_div(num_cells, threadblock_size);

    // initialise state values directly on the device
    init_state_values<<<threadblocks, threadblock_size, 0, stream>>>(d_states, num_cells,
                                                                     padded_num_cells);

    // initialise parameter values on the host and copy to the device
    init_parameters_values(h_parameters);
    cudaMemcpyAsync(d_parameters, h_parameters, parameters_size, cudaMemcpyHostToDevice, stream);

    // synchonize the stream to ensure that the init kernel has completer before
    // starting the timer and entering the main loop
    cudaStreamSynchronize(stream);
    switch (scheme) {
    case scheme_arg_FE:
        printf("Solving with the FE scheme\n");
        break;
    case scheme_arg_RL:
        printf("Solving with the RL scheme\n");
        break;
    case scheme_arg_GRL1:
        printf("Solving with the GRL1 scheme\n");
        break;
    default:
        fprintf(stderr, "ERROR: Unsupported scheme!\n");
        exit(EXIT_FAILURE);
    }
    auto t1_solve = std::chrono::high_resolution_clock::now();

    double t = 0;
    for (int i = 0; i < num_timesteps; i++) {
        switch (scheme) {
        case scheme_arg_FE:
            step_FE<<<threadblocks, threadblock_size, 0, stream>>>(d_states, t, dt, d_parameters,
                                                                   num_cells, padded_num_cells);
            break;

        case scheme_arg_RL:
            step_RL<<<threadblocks, threadblock_size, 0, stream>>>(d_states, t, dt, d_parameters,
                                                                   num_cells, padded_num_cells);
            break;

        case scheme_arg_GRL1:
            step_GRL1<<<threadblocks, threadblock_size, 0, stream>>>(d_states, t, dt, d_parameters,
                                                                     num_cells, padded_num_cells);
            break;
        }

        t += dt;
    }
    cudaStreamSynchronize(stream);
    auto t2_solve = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur(t2_solve - t1_solve);
    double solution_time = dur.count();
    double cell_steps_per_second = num_cells * (double) num_timesteps / solution_time;
    printf("Solved %d timesteps with %d cells in %g seconds.\n", num_timesteps, num_cells,
           solution_time);
    printf("Throughput: %g cell steps / second.\n", cell_steps_per_second);

    // Copy back the state values and check the transmembrane potential (V)
    cudaMemcpyAsync(h_states, d_states, states_size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    double min_V = h_states[STATE_V * padded_num_cells];
    double max_V = h_states[STATE_V * padded_num_cells];
    for (int i = 0; i < num_cells; i++) {
        double V = h_states[STATE_V * padded_num_cells + i];
        if (V < min_V) {
            min_V = V;
        }
        if (V > max_V) {
            max_V = V;
        }
    }
    printf("Final V values are in the interval [%.17e, %.17e]\n", min_V, max_V);

    // Release memory and resources
    cudaStreamDestroy(stream);

    cudaFree(d_states);
    cudaFree(d_parameters);
    cudaFreeHost(h_states);
    cudaFreeHost(h_parameters);
}

int main(int argc, char *argv[])
{
    struct gengetopt_args_info args_info;
    if (cmdline_parser(argc, argv, &args_info) != 0) {
        return EXIT_FAILURE;
    }

    int num_cells = (int) args_info.num_cells_arg;
    int num_timesteps = args_info.num_timesteps_arg;
    double dt = args_info.dt_arg;
    enum enum_scheme scheme = args_info.scheme_arg;
    int device_id = args_info.device_id_arg;

    cmdline_parser_free(&args_info);

    bench(num_cells, num_timesteps, dt, scheme, device_id);

    return 0;
}
