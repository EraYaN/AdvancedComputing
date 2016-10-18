#include "infoli.h"

void ComputeOneCell(cellCompParams *cellCompParamsPtr){

    //The three compartments can be computed concurrently but only across a single sim step
    CompDend(cellCompParamsPtr);
    CompSoma(cellCompParamsPtr);
    CompAxon(cellCompParamsPtr);

    return;
}

void CompDend(cellCompParams *cellCompParamsPtr){

    struct channelParams chPrms;
    struct dendCurrVoltPrms chComps;

    //printf("Dendrite ");

    //Prepare pointers to inputs/outputs
    chPrms.v = &cellCompParamsPtr->prevCellState->dend.V_dend;
    chPrms.prevComp1 = &cellCompParamsPtr->prevCellState->dend.Hcurrent_q;
    chPrms.newComp1 = &cellCompParamsPtr->newCellState->dend.Hcurrent_q;
    //Compute
    DendHCurr(&chPrms);

    //Prepare pointers to inputs/outputs
    chPrms.v = &cellCompParamsPtr->prevCellState->dend.V_dend;
    chPrms.prevComp1 = &cellCompParamsPtr->prevCellState->dend.Calcium_r;
    chPrms.newComp1 = &cellCompParamsPtr->newCellState->dend.Calcium_r;
    //Compute
    DendCaCurr(&chPrms);

    //Prepare pointers to inputs/outputs
    chPrms.prevComp1 = &cellCompParamsPtr->prevCellState->dend.Potassium_s;
    chPrms.prevComp2 = &cellCompParamsPtr->prevCellState->dend.Ca2Plus;
    chPrms.newComp1 = &cellCompParamsPtr->newCellState->dend.Potassium_s;
    //Compute
    DendKCurr(&chPrms);

    //Prepare pointers to inputs/outputs
    chPrms.prevComp1 = &cellCompParamsPtr->prevCellState->dend.Ca2Plus;
    chPrms.prevComp2 = &cellCompParamsPtr->prevCellState->dend.I_CaH;
    chPrms.newComp1 = &cellCompParamsPtr->newCellState->dend.Ca2Plus;
    //Compute
    DendCal(&chPrms);

    chComps.iC = IcNeighbors(cellCompParamsPtr->neighVdend, cellCompParamsPtr->prevCellState->dend.V_dend);
    chComps.iApp = &cellCompParamsPtr->iAppIn;
    chComps.vDend = &cellCompParamsPtr->prevCellState->dend.V_dend;
    chComps.newVDend = &cellCompParamsPtr->newCellState->dend.V_dend;
    chComps.vSoma = &cellCompParamsPtr->prevCellState->soma.V_soma;
    chComps.q = &cellCompParamsPtr->newCellState->dend.Hcurrent_q;
    chComps.r = &cellCompParamsPtr->newCellState->dend.Calcium_r;
    chComps.s = &cellCompParamsPtr->newCellState->dend.Potassium_s;
    chComps.newI_CaH = &cellCompParamsPtr->newCellState->dend.I_CaH;
    DendCurrVolt(&chComps);

    return;
}

void DendHCurr(struct channelParams *chPrms){

    user_float_t q_inf, tau_q, dq_dt, q_local;

    //Get inputs
    user_float_t prevV_dend = *chPrms->v;
    user_float_t prevHcurrent_q = *chPrms->prevComp1;

    // Update dendritic H current component
    q_inf = 1 /(1 + exp((prevV_dend + 80) / 4));
    tau_q = 1 /(exp(-0.086 * prevV_dend - 14.6) + exp(0.070 * prevV_dend - 1.87));
    dq_dt = (q_inf - prevHcurrent_q) / tau_q;
    q_local = DELTA * dq_dt + prevHcurrent_q;
    //Put result
    *chPrms->newComp1 = q_local;

    return;
}
void DendCaCurr(struct channelParams *chPrms){

    user_float_t alpha_r, beta_r, r_inf, tau_r, dr_dt, r_local;

    //Get inputs
    user_float_t prevV_dend = *chPrms->v;
    user_float_t prevCalcium_r = *chPrms->prevComp1;

    // Update dendritic high-threshold Ca current component
    alpha_r = 1.7 / (1 + exp( -(prevV_dend - 5) / 13.9));
    beta_r = 0.02 * (prevV_dend + 8.5) / (exp((prevV_dend + 8.5) / 5) - 1);
    r_inf = alpha_r / (alpha_r + beta_r);
    tau_r = 5 / (alpha_r + beta_r);
    dr_dt = (r_inf - prevCalcium_r) / tau_r;
    r_local = DELTA * dr_dt + prevCalcium_r;
    //Put result
    *chPrms->newComp1 = r_local;

    return;
}
void DendKCurr(struct channelParams *chPrms){

    user_float_t  alpha_s, beta_s, s_inf, tau_s, ds_dt, s_local;

    //Get inputs
    user_float_t prevPotassium_s = *chPrms->prevComp1;
    user_float_t prevCa2Plus = *chPrms->prevComp2;

    // Update dendritic Ca-dependent K current component
    alpha_s = min((0.00002*prevCa2Plus), 0.01);
    beta_s = 0.015;
    s_inf = alpha_s / (alpha_s + beta_s);
    tau_s = 1 / (alpha_s + beta_s);
    ds_dt = (s_inf - prevPotassium_s) / tau_s;
    s_local = DELTA * ds_dt + prevPotassium_s;
    //Put result
    *chPrms->newComp1 = s_local;

    return;
}
//Consider merging DendCal into DendKCurr since DendCal's output doesn't go to DendCurrVolt but to DendKCurr
void DendCal(struct channelParams *chPrms){

    user_float_t  dCa_dt, Ca2Plus_local;

    //Get inputs
    user_float_t prevCa2Plus = *chPrms->prevComp1;
    user_float_t prevI_CaH = *chPrms->prevComp2;

    // update Calcium concentration
    dCa_dt = -3 * prevI_CaH - 0.075 * prevCa2Plus;
    Ca2Plus_local = DELTA * dCa_dt + prevCa2Plus;
    //Put result
    *chPrms->newComp1 = Ca2Plus_local;//This state value is read in DendKCurr

    return;
}

void DendCurrVolt(struct dendCurrVoltPrms *chComps){

    //Loca variables
    user_float_t I_sd, I_CaH, I_K_Ca, I_ld, I_h, dVd_dt;

    //Get inputs
    user_float_t I_c = chComps->iC;
    user_float_t I_app = *chComps->iApp;
    user_float_t prevV_dend = *chComps->vDend;
    user_float_t prevV_soma = *chComps->vSoma;
    user_float_t q = *chComps->q;
    user_float_t r = *chComps->r;
    user_float_t s = *chComps->s;

    // DENDRITIC CURRENTS

    // Soma-dendrite interaction current I_sd
    I_sd   = (G_INT / (1 - P1)) * (prevV_dend - prevV_soma);
    // Inward high-threshold Ca current I_CaH
    I_CaH  =  G_CAH * r * r * (prevV_dend - V_CA);
    // Outward Ca-dependent K current I_K_Ca
    I_K_Ca =  G_K_CA * s * (prevV_dend - V_K);
    // Leakage current I_ld
    I_ld   =  G_LD * (prevV_dend - V_L);
    // Inward anomalous rectifier I_h
    I_h    =  G_H * q * (prevV_dend - V_H);

    dVd_dt = (-(I_CaH   + I_sd  + I_ld + I_K_Ca + I_c + I_h) + I_app) / C_M;

    //Put result (update V_dend)
    *chComps->newVDend = DELTA * dVd_dt + prevV_dend;
    *chComps->newI_CaH = I_CaH;//This is a state value read in DendCal
    return;
}
user_float_t IcNeighbors(user_float_t *neighVdend, user_float_t prevV_dend){

    int i;
    user_float_t f, V, I_c;
    //printf("Ic[0]= %f\n", neighVdend[0]);

    I_c = 0;
    for(i=0;i<8;i++){
        V = prevV_dend - neighVdend[i];
        f = 0.8 * exp(-1*pow(V, 2)/100) + 0.2;    // SCHWEIGHOFER 2004 VERSION
        I_c = I_c + (CONDUCTANCE * f * V);
    }

    return I_c;
}

void CompSoma(cellCompParams *cellCompParamsPtr){

    struct channelParams chPrms;
    struct somaCurrVoltPrms chComps;

    // update somatic components
    // SCHWEIGHOFER:

    //Prepare pointers to inputs/outputs
    chPrms.v = &cellCompParamsPtr->prevCellState->soma.V_soma;
    chPrms.prevComp1 = &cellCompParamsPtr->prevCellState->soma.Calcium_k;
    chPrms.prevComp2 = &cellCompParamsPtr->prevCellState->soma.Calcium_l;
    chPrms.newComp1 = &cellCompParamsPtr->newCellState->soma.Calcium_k;
    chPrms.newComp2 = &cellCompParamsPtr->newCellState->soma.Calcium_l;
    //Compute
    SomaCalcium(&chPrms);

    //Prepare pointers to inputs/outputs
    chPrms.v = &cellCompParamsPtr->prevCellState->soma.V_soma;
    chPrms.prevComp1 = &cellCompParamsPtr->prevCellState->soma.Sodium_m;
    chPrms.prevComp2 = &cellCompParamsPtr->prevCellState->soma.Sodium_h;
    chPrms.newComp1 = &cellCompParamsPtr->newCellState->soma.Sodium_m;
    chPrms.newComp2 = &cellCompParamsPtr->newCellState->soma.Sodium_h;
    //Compute
    SomaSodium(&chPrms);

    //Prepare pointers to inputs/outputs
    chPrms.v = &cellCompParamsPtr->prevCellState->soma.V_soma;
    chPrms.prevComp1 = &cellCompParamsPtr->prevCellState->soma.Potassium_n;
    chPrms.prevComp2 = &cellCompParamsPtr->prevCellState->soma.Potassium_p;
    chPrms.newComp1 = &cellCompParamsPtr->newCellState->soma.Potassium_n;
    chPrms.newComp2 = &cellCompParamsPtr->newCellState->soma.Potassium_p;
    //Compute
    SomaPotassium(&chPrms);

    //Prepare pointers to inputs/outputs
    chPrms.v = &cellCompParamsPtr->prevCellState->soma.V_soma;
    chPrms.prevComp1 = &cellCompParamsPtr->prevCellState->soma.Potassium_x_s;
    chPrms.newComp1 = &cellCompParamsPtr->newCellState->soma.Potassium_x_s;
    //Compute
    SomaPotassiumX(&chPrms);

    chComps.g_CaL = &cellCompParamsPtr->prevCellState->soma.g_CaL;
    chComps.vDend = &cellCompParamsPtr->prevCellState->dend.V_dend;
    chComps.vSoma = &cellCompParamsPtr->prevCellState->soma.V_soma;
    chComps.newVSoma = &cellCompParamsPtr->newCellState->soma.V_soma;
    chComps.vAxon = &cellCompParamsPtr->prevCellState->axon.V_axon;
    chComps.k = &cellCompParamsPtr->newCellState->soma.Calcium_k;
    chComps.l = &cellCompParamsPtr->newCellState->soma.Calcium_l;
    chComps.m = &cellCompParamsPtr->newCellState->soma.Sodium_m;
    chComps.h = &cellCompParamsPtr->newCellState->soma.Sodium_h;
    chComps.n = &cellCompParamsPtr->newCellState->soma.Potassium_n;
    chComps.x_s = &cellCompParamsPtr->newCellState->soma.Potassium_x_s;
    SomaCurrVolt(&chComps);

    return;
}
void SomaCalcium(struct channelParams *chPrms){

    user_float_t k_inf, l_inf, tau_k, tau_l, dk_dt, dl_dt, k_local, l_local;

    //Get inputs
    user_float_t prevV_soma = *chPrms->v;
    user_float_t prevCalcium_k = *chPrms->prevComp1;
    user_float_t prevCalcium_l = *chPrms->prevComp2;

    k_inf = (1 / (1 + exp(-1 * (prevV_soma + 61)   / 4.2)));
    l_inf = (1 / (1 + exp((     prevV_soma + 85.5) / 8.5)));
    tau_k = 1;
    tau_l = ((20 * exp((prevV_soma + 160) / 30) / (1 + exp((prevV_soma + 84) / 7.3))) +35);
    dk_dt = (k_inf - prevCalcium_k) / tau_k;
    dl_dt = (l_inf - prevCalcium_l) / tau_l;
    k_local = DELTA * dk_dt + prevCalcium_k;
    l_local = DELTA * dl_dt + prevCalcium_l;
    //Put result
    *chPrms->newComp1= k_local;
    *chPrms->newComp2= l_local;

    return;
}
void SomaSodium(struct channelParams *chPrms){

    user_float_t m_inf, h_inf, tau_h, dh_dt, m_local, h_local;

    //Get inputs
    user_float_t prevV_soma = *chPrms->v;
    //user_float_t prevSodium_m = *chPrms->prevComp1;
    user_float_t prevSodium_h = *chPrms->prevComp2;

    // RAT THALAMOCORTICAL SODIUM:
    m_inf   = 1 / (1 + (exp((-30 - prevV_soma)/ 5.5)));
    h_inf   = 1 / (1 + (exp((-70 - prevV_soma)/-5.8)));
    tau_h   =       3 * exp((-40 - prevV_soma)/33);
    dh_dt   = (h_inf - prevSodium_h)/tau_h;
    m_local       = m_inf;
    h_local       = prevSodium_h + DELTA * dh_dt;
    //Put result
    *chPrms->newComp1 = m_local;
    *chPrms->newComp2 = h_local;

    return;
}
void SomaPotassium(struct channelParams *chPrms){

    user_float_t n_inf, p_inf, tau_n, tau_p, dn_dt, dp_dt, n_local, p_local;

    //Get inputs
    user_float_t prevV_soma = *chPrms->v;
    user_float_t prevPotassium_n = *chPrms->prevComp1;
    user_float_t prevPotassium_p = *chPrms->prevComp2;

    // NEOCORTICAL
    n_inf = 1 / (1 + exp( ( -3 - prevV_soma) /  10));
    p_inf = 1 / (1 + exp( (-51 - prevV_soma) / -12));
    tau_n =   5 + (  47 * exp( -(-50 - prevV_soma) /  900));
    tau_p = tau_n;
    dn_dt = (n_inf - prevPotassium_n) / tau_n;
    dp_dt = (p_inf - prevPotassium_p) / tau_p;
    n_local = DELTA * dn_dt + prevPotassium_n;
    p_local = DELTA * dp_dt + prevPotassium_p;
    //Put result
    *chPrms->newComp1 = n_local;
    *chPrms->newComp2 = p_local;

    return;
}
void SomaPotassiumX(struct channelParams *chPrms){

    user_float_t alpha_x_s, beta_x_s, x_inf_s, tau_x_s, dx_dt_s, x_s_local;

    //Get inputs
    user_float_t prevV_soma = *chPrms->v;
    user_float_t prevPotassium_x_s = *chPrms->prevComp1;

    // Voltage-dependent (fast) potassium
    alpha_x_s = 0.13 * (prevV_soma + 25) / (1 - exp(-(prevV_soma + 25) / 10));
    beta_x_s  = 1.69 * exp(-0.0125 * (prevV_soma + 35));
    x_inf_s   = alpha_x_s / (alpha_x_s + beta_x_s);
    tau_x_s   =         1 / (alpha_x_s + beta_x_s);
    dx_dt_s   = (x_inf_s - prevPotassium_x_s) / tau_x_s;
    x_s_local       = 0.05 * dx_dt_s + prevPotassium_x_s;
    //Put result
    *chPrms->newComp1 = x_s_local;

    return;
}
void SomaCurrVolt(struct somaCurrVoltPrms *chComps){

    //Local variables
    user_float_t I_ds, I_CaL, I_Na_s, I_ls, I_Kdr_s, I_K_s, I_as, dVs_dt;

    //Get inputs
    user_float_t g_CaL = *chComps->g_CaL;
    user_float_t prevV_dend = *chComps->vDend;
    user_float_t prevV_soma = *chComps->vSoma;
    user_float_t prevV_axon = *chComps->vAxon;
    user_float_t k = *chComps->k;
    user_float_t l = *chComps->l;
    user_float_t m = *chComps->m;
    user_float_t h = *chComps->h;
    user_float_t n = *chComps->n;
    user_float_t x_s = *chComps->x_s;

    // SOMATIC CURRENTS

    // Dendrite-soma interaction current I_ds
    I_ds  = (G_INT / P1) * (prevV_soma - prevV_dend);
    // Inward low-threshold Ca current I_CaL
    I_CaL = g_CaL * k * k * k * l * (prevV_soma - V_CA); //k^3
    // Inward Na current I_Na_s
    I_Na_s  = G_NA_S * m * m * m * h * (prevV_soma - V_NA);
    // Leakage current I_ls
    I_ls  = G_LS * (prevV_soma - V_L);
    // Outward delayed potassium current I_Kdr
    I_Kdr_s = G_KDR_S * n * n * n * n * (prevV_soma - V_K); // SCHWEIGHOFER
    // I_K_s
    I_K_s   = G_K_S * pow(x_s, 4) * (prevV_soma - V_K);
    // Axon-soma interaction current I_as
    I_as    = (G_INT / (1 - P2)) * (prevV_soma - prevV_axon);

    dVs_dt = (-(I_CaL   + I_ds  + I_as + I_Na_s + I_ls   + I_Kdr_s + I_K_s)) / C_M;
    *chComps->newVSoma = DELTA * dVs_dt + prevV_soma;

    return;
}
void CompAxon(cellCompParams *cellCompParamsPtr){

    struct channelParams chPrms;
    struct axonCurrVoltPrms chComps;

    // update somatic components
    // SCHWEIGHOFER:

    //Prepare pointers to inputs/outputs
    chPrms.v = &cellCompParamsPtr->prevCellState->axon.V_axon;
    chPrms.prevComp1 = &cellCompParamsPtr->prevCellState->axon.Sodium_h_a;
    chPrms.newComp1 = &cellCompParamsPtr->newCellState->axon.Sodium_h_a;
    chPrms.newComp2 = &cellCompParamsPtr->newCellState->axon.Sodium_m_a;
    //Compute
    AxonSodium(&chPrms);

    //Prepare pointers to inputs/outputs
    chPrms.v = &cellCompParamsPtr->prevCellState->axon.V_axon;
    chPrms.prevComp1 = &cellCompParamsPtr->prevCellState->axon.Potassium_x_a;
    chPrms.newComp1 = &cellCompParamsPtr->newCellState->axon.Potassium_x_a;
    //Compute
    AxonPotassium(&chPrms);

    //Get inputs
    chComps.vSoma = &cellCompParamsPtr->prevCellState->soma.V_soma;
    chComps.vAxon = &cellCompParamsPtr->prevCellState->axon.V_axon;
    chComps.newVAxon = &cellCompParamsPtr->newCellState->axon.V_axon;
    chComps.m_a = &cellCompParamsPtr->newCellState->axon.Sodium_m_a;
    chComps.h_a = &cellCompParamsPtr->newCellState->axon.Sodium_h_a;
    chComps.x_a = &cellCompParamsPtr->newCellState->axon.Potassium_x_a;
    AxonCurrVolt(&chComps);

    return;
}

void AxonSodium(struct channelParams *chPrms){

    user_float_t m_inf_a, h_inf_a, tau_h_a, dh_dt_a, m_a_local, h_a_local;

    //Get inputs
    user_float_t prevV_axon = *chPrms->v;
    user_float_t prevSodium_h_a = *chPrms->prevComp1;

    // Update axonal Na components
    // NOTE: current has shortened inactivation to account for high
    // firing frequencies in axon hillock
    m_inf_a   = 1 / (1 + (exp((-30 - prevV_axon)/ 5.5)));
    h_inf_a   = 1 / (1 + (exp((-60 - prevV_axon)/-5.8)));
    tau_h_a   =     1.5 * exp((-40 - prevV_axon)/33);
    dh_dt_a   = (h_inf_a - prevSodium_h_a)/tau_h_a;
    m_a_local = m_inf_a;
    h_a_local = prevSodium_h_a + DELTA * dh_dt_a;
    //Put result
    *chPrms->newComp1 = h_a_local;
    *chPrms->newComp2 = m_a_local;

    return;
}
void AxonPotassium(struct channelParams *chPrms){

    user_float_t alpha_x_a, beta_x_a, x_inf_a, tau_x_a, dx_dt_a, x_a_local;

    //Get inputs
    user_float_t prevV_axon = *chPrms->v;
    user_float_t prevPotassium_x_a = *chPrms->prevComp1;

    // D'ANGELO 2001 -- Voltage-dependent potassium
    alpha_x_a = 0.13 * (prevV_axon + 25) / (1 - exp(-(prevV_axon + 25) / 10));
    beta_x_a  = 1.69 * exp(-0.0125 * (prevV_axon + 35));
    x_inf_a   = alpha_x_a / (alpha_x_a + beta_x_a);
    tau_x_a   =         1 / (alpha_x_a + beta_x_a);
    dx_dt_a   = (x_inf_a - prevPotassium_x_a) / tau_x_a;
    x_a_local = 0.05 * dx_dt_a + prevPotassium_x_a;
    //Put result
    *chPrms->newComp1 = x_a_local;

    return;
}
void AxonCurrVolt(struct axonCurrVoltPrms *chComps){

    //Local variable
    user_float_t I_Na_a, I_la, I_sa, I_K_a, dVa_dt;

    //Get inputs
    user_float_t prevV_soma = *chComps->vSoma;
    user_float_t prevV_axon = *chComps->vAxon;
    user_float_t m_a = *chComps->m_a;
    user_float_t h_a = *chComps->h_a;
    user_float_t x_a = *chComps->x_a;

    // AXONAL CURRENTS
    // Sodium
    I_Na_a  = G_NA_A  * m_a * m_a * m_a * h_a * (prevV_axon - V_NA);
    // Leak
    I_la    = G_LA    * (prevV_axon - V_L);
    // Soma-axon interaction current I_sa
    I_sa    = (G_INT / P2) * (prevV_axon - prevV_soma);
    // Potassium (transient)
    I_K_a   = G_K_A * pow(x_a, 4) * (prevV_axon - V_K);
    dVa_dt = (-(I_K_a + I_sa + I_la + I_Na_a)) / C_M;
    *chComps->newVAxon = DELTA * dVa_dt + prevV_axon;

    return;
}
