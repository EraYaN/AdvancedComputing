#include "kernel.h"

void printComParams(user_float_t *cellCompParamsPtr) {
	for (int i = 0; i < STATEADD;i++)
		printf("Neighbour %d: %f\n", i, cellCompParamsPtr[VNEIGHSTARTADD + i]);

	for (int i = 0; i < STATE_SIZE;i++)
		printf("Prev State %d: %f\n", i, cellCompParamsPtr[PREVSTATESTARTADD + i]);

	for (int i = 0; i < STATE_SIZE;i++)
		printf("Next State %d: %f\n", i, cellCompParamsPtr[NEXTSTATESTARTADD + i]);

}

//#include <math.h>
void ComputeOneCell(user_float_t *cellCompParamsPtr) {

	//The three compartments can be computed concurrently but only across a single sim step

	CompDend(cellCompParamsPtr);
	CompSoma(cellCompParamsPtr);
	CompAxon(cellCompParamsPtr);

	return;
}

void CompDend(user_float_t *cellCompParamsPtr) {

	/*for (int ind = 0; ind < LOCAL_PARAM_SIZE; ind++)
		printf("%d: %lf\n", ind,cellCompParamsPtr[ind]);*/

	user_float_t *chPrms_v;
	user_float_t *chPrms_prevComp1, *chPrms_prevComp2;
	user_float_t *chPrms_newComp1;// *chPrms_newComp2;
	user_float_t *chComps_iApp;
	user_float_t chComps_iC;
	user_float_t *chComps_vDend;
	user_float_t *chComps_vSoma;
	user_float_t *chComps_q, *chComps_r, *chComps_s;
	user_float_t *chComps_newVDend;
	user_float_t *chComps_newI_CaH;

	//printf("Dendrite ");

	//Prepare pointers to inputs/outputs
	chPrms_v = &(cellCompParamsPtr[PREVSTATESTARTADD + DEND_V]); //&cellCompParamsPtr->prevCellState->dend.V_dend;
	chPrms_prevComp1 = &(cellCompParamsPtr[PREVSTATESTARTADD + DEND_H]); //&cellCompParamsPtr->prevCellState->dend.Hcurrent_q;
	chPrms_newComp1 = &(cellCompParamsPtr[NEXTSTATESTARTADD + DEND_H]); //&cellCompParamsPtr->newCellState->dend.Hcurrent_q;
																		//Compute
	DendHCurr(chPrms_v, chPrms_prevComp1, chPrms_newComp1);

	//printf("VDend (DendHCurr): %lf\n", chPrms_v);
	//Prepare pointers to inputs/outputs
	chPrms_v = &(cellCompParamsPtr[PREVSTATESTARTADD + DEND_V]);  //&cellCompParamsPtr->prevCellState->dend.V_dend;
	chPrms_prevComp1 = &(cellCompParamsPtr[PREVSTATESTARTADD + DEND_CAL]); //&cellCompParamsPtr->prevCellState->dend.Calcium_r;
	chPrms_newComp1 = &(cellCompParamsPtr[NEXTSTATESTARTADD + DEND_CAL]); //&cellCompParamsPtr->newCellState->dend.Calcium_r;
																	  //Compute
	DendCaCurr(chPrms_v, chPrms_prevComp1, chPrms_newComp1);

	//printf("VDend (DendCaCurr): %lf\n", chPrms_v);
	//Prepare pointers to inputs/outputs
	chPrms_prevComp1 = &(cellCompParamsPtr[PREVSTATESTARTADD + DEND_P]); //&cellCompParamsPtr->prevCellState->dend.Potassium_s;
	chPrms_prevComp2 = &(cellCompParamsPtr[PREVSTATESTARTADD + DEND_CA2]); //&cellCompParamsPtr->prevCellState->dend.Ca2Plus;
	chPrms_newComp1 = &(cellCompParamsPtr[NEXTSTATESTARTADD + DEND_P]);  //&cellCompParamsPtr->newCellState->dend.Potassium_s;
																		 //Compute
	DendKCurr(chPrms_prevComp1, chPrms_prevComp2, chPrms_newComp1);

	//Prepare pointers to inputs/outputs
	chPrms_prevComp1 = &(cellCompParamsPtr[PREVSTATESTARTADD + DEND_CA2]); //&cellCompParamsPtr->prevCellState->dend.Ca2Plus;
	chPrms_prevComp2 = &(cellCompParamsPtr[PREVSTATESTARTADD + DEND_I]); //&cellCompParamsPtr->prevCellState->dend.I_CaH;
	chPrms_newComp1 = &(cellCompParamsPtr[NEXTSTATESTARTADD + DEND_CA2]);  //&cellCompParamsPtr->newCellState->dend.Ca2Plus;
																		   //Compute
	DendCal(chPrms_prevComp1, chPrms_prevComp2, chPrms_newComp1);

	chComps_iC = IcNeighbors(cellCompParamsPtr, cellCompParamsPtr[PREVSTATESTARTADD + DEND_V]);
	//IcNeighbors(cellCompParamsPtr->neighVdend, cellCompParamsPtr->prevCellState->dend.V_dend);
	chComps_iApp = &(cellCompParamsPtr[0]); //&cellCompParamsPtr->iAppIn;
	chComps_vDend = &(cellCompParamsPtr[PREVSTATESTARTADD]); //&cellCompParamsPtr->prevCellState->dend.V_dend;
	chComps_newVDend = &(cellCompParamsPtr[NEXTSTATESTARTADD + DEND_V]); //&cellCompParamsPtr->newCellState->dend.V_dend;
	chComps_vSoma = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_V]); //&cellCompParamsPtr->prevCellState->soma.V_soma
	chComps_q = &(cellCompParamsPtr[NEXTSTATESTARTADD + DEND_H]); // &cellCompParamsPtr->newCellState->dend.Hcurrent_q;
	chComps_r = &(cellCompParamsPtr[NEXTSTATESTARTADD + DEND_CAL]); //&cellCompParamsPtr->newCellState->dend.Calcium_r;
	chComps_s = &(cellCompParamsPtr[NEXTSTATESTARTADD + DEND_P]); //&cellCompParamsPtr->newCellState->dend.Potassium_s;
	chComps_newI_CaH = &(cellCompParamsPtr[NEXTSTATESTARTADD + DEND_I]); //&cellCompParamsPtr->newCellState->dend.I_CaH;
	DendCurrVolt(chComps_iC, chComps_iApp, chComps_vDend, chComps_newVDend, chComps_vSoma, chComps_q, chComps_r, chComps_s, chComps_newI_CaH);
	//printf("%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\n", chComps_iC, *chComps_iApp, *chComps_vDend, *chComps_newVDend, *chComps_vSoma, *chComps_q, *chComps_r, *chComps_s, *chComps_newI_CaH);
	//printf("VDend (DendCurrVolt): %lf\n", chComps_newVDend);

	return;
}

void DendHCurr(user_float_t *chPrms_v, user_float_t *chPrms_prevComp1, user_float_t *chPrms_newComp1) {

	user_float_t q_inf, tau_q, dq_dt, q_local;

	//Get inputs
	user_float_t prevV_dend = *chPrms_v; // *chPrms->v;
	user_float_t prevHcurrent_q = *chPrms_prevComp1;//*chPrms->prevComp1;

	// Update dendritic H current component
	q_inf = 1 / (1 + exp((prevV_dend + 80) / 4));
	tau_q = 1 / (exp(-0.086 * prevV_dend - 14.6) + exp(0.070 * prevV_dend - 1.87));
	dq_dt = (q_inf - prevHcurrent_q) / tau_q;
	q_local = DELTA * dq_dt + prevHcurrent_q;
	//Put result
	*chPrms_newComp1 = q_local;

	return;
}

void DendCaCurr(user_float_t *chPrms_v, user_float_t *chPrms_prevComp1, user_float_t *chPrms_newComp1) {

	user_float_t alpha_r, beta_r, r_inf, tau_r, dr_dt, r_local;

	//Get inputs
	user_float_t prevV_dend = *chPrms_v; //*chPrms->v;
	user_float_t prevCalcium_r = *chPrms_prevComp1; //*chPrms->prevComp1;

	// Update dendritic high-threshold Ca current component
	alpha_r = 1.7 / (1 + exp(-(prevV_dend - 5) / 13.9));
	beta_r = 0.02 * (prevV_dend + 8.5) / (exp((prevV_dend + 8.5) / 5) - 1);
	r_inf = alpha_r / (alpha_r + beta_r);
	tau_r = 5 / (alpha_r + beta_r);
	dr_dt = (r_inf - prevCalcium_r) / tau_r;
	r_local = DELTA * dr_dt + prevCalcium_r;
	//Put result
	*chPrms_newComp1 = r_local; // *chPrms->newComp1

	return;
}
void DendKCurr(user_float_t *chPrms_prevComp1, user_float_t *chPrms_prevComp2, user_float_t *chPrms_newComp1) {

	user_float_t  alpha_s = 0.01, beta_s, s_inf, tau_s, ds_dt, s_local;

	//Get inputs
	user_float_t prevPotassium_s = *chPrms_prevComp1;//*chPrms->prevComp1;
	user_float_t prevCa2Plus = *chPrms_prevComp2; //*chPrms->prevComp2;

	// Update dendritic Ca-dependent K current component
	if ((0.00002*prevCa2Plus) < 0.01)
		alpha_s = (0.00002*prevCa2Plus);
	beta_s = 0.015;
	s_inf = alpha_s / (alpha_s + beta_s);
	tau_s = 1 / (alpha_s + beta_s);
	ds_dt = (s_inf - prevPotassium_s) / tau_s;
	s_local = DELTA * ds_dt + prevPotassium_s;
	//Put result
	*chPrms_newComp1 = s_local; //*chPrms->newComp1

	return;
}
//Consider merging DendCal into DendKCurr since DendCal's output doesn't go to DendCurrVolt but to DendKCurr
void DendCal(user_float_t *chPrms_prevComp1, user_float_t *chPrms_prevComp2, user_float_t *chPrms_newComp1) {

	user_float_t  dCa_dt, Ca2Plus_local;

	//Get inputs
	user_float_t prevCa2Plus = *chPrms_prevComp1; //*chPrms->prevComp1;
	user_float_t prevI_CaH = *chPrms_prevComp2; //*chPrms->prevComp2;

	// update Calcium concentration
	dCa_dt = -3 * prevI_CaH - 0.075 * prevCa2Plus;
	Ca2Plus_local = DELTA * dCa_dt + prevCa2Plus;
	//Put result
	*chPrms_newComp1 = Ca2Plus_local; //*chPrms->newComp1 //This state value is read in DendKCurr

	return;
}

void DendCurrVolt(user_float_t chComps_iC, user_float_t *chComps_iApp, user_float_t *chComps_vDend, user_float_t *chComps_newVDend, user_float_t *chComps_vSoma, user_float_t *chComps_q, user_float_t *chComps_r, user_float_t *chComps_s, user_float_t *chComps_newI_CaH) {

	//Local variables
	user_float_t I_sd, I_CaH, I_K_Ca, I_ld, I_h, dVd_dt;

	//Get inputs
	user_float_t I_c = chComps_iC; //chComps->iC;
	user_float_t I_app = *chComps_iApp; //*chComps->iApp;
	user_float_t prevV_dend = *chComps_vDend; //*chComps->vDend;
	user_float_t prevV_soma = *chComps_vSoma; //*chComps->vSoma;
	user_float_t q = *chComps_q; //*chComps->q;
	user_float_t r = *chComps_r; //*chComps->r;
	user_float_t s = *chComps_s; //*chComps->s;

	// DENDRITIC CURRENTS

	// Soma-dendrite interaction current I_sd
	I_sd = (G_INT / (1 - P1)) * (prevV_dend - prevV_soma);
	// Inward high-threshold Ca current I_CaH
	I_CaH = G_CAH * r * r * (prevV_dend - V_CA);
	// Outward Ca-dependent K current I_K_Ca
	I_K_Ca = G_K_CA * s * (prevV_dend - V_K);
	// Leakage current I_ld
	I_ld = G_LD * (prevV_dend - V_L);
	// Inward anomalous rectifier I_h
	I_h = G_H * q * (prevV_dend - V_H);

	dVd_dt = (-(I_CaH + I_sd + I_ld + I_K_Ca + I_c + I_h) + I_app) / C_M;

	//Put result (update V_dend)
	*chComps_newVDend = DELTA * dVd_dt + prevV_dend; //*chComps->newVDend
	*chComps_newI_CaH = I_CaH; //*chComps->newI_CaH //This is a state value read in DendCal
	return;
}
user_float_t IcNeighbors(user_float_t *neighVdend, user_float_t prevV_dend) {

	int i;
	user_float_t f, V, I_c;
	//printf("Ic[0]= %f\n", neighVdend[0]);

	I_c = 0;
#pragma unroll 8
	for (i = 0;i < 8;i++) {
		//printf("%d prevdend: %0.10lf, neighVdend: %0.10lf\n",i, prevV_dend, *neighVdend );
		V = prevV_dend - neighVdend[VNEIGHSTARTADD + i];
		f = 0.8 * exp(-1 * pow(V, 2) / 100) + 0.2;    // SCHWEIGHOFER 2004 VERSION
		I_c = I_c + (CONDUCTANCE * f * V);
	}
	//printf("ja hallo hier is IC, met wie spreek ik: %0.10lf\n", I_c);
	return I_c;
}

void CompSoma(user_float_t *cellCompParamsPtr) {

	user_float_t *chPrms_v;
	user_float_t *chPrms_prevComp1, *chPrms_prevComp2;
	user_float_t *chPrms_newComp1, *chPrms_newComp2;
	user_float_t *chComps_g_CaL;
	user_float_t *chComps_vSoma;
	user_float_t *chComps_vDend;
	user_float_t *chComps_vAxon;
	user_float_t *chComps_k, *chComps_l, *chComps_m, *chComps_h, *chComps_n, *chComps_x_s;
	user_float_t *chComps_newVSoma;

	// update somatic components
	// SCHWEIGHOFER:

	//Prepare pointers to inputs/outputs
	chPrms_v = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_V]); //&cellCompParamsPtr->prevCellState->soma.V_soma;
	chPrms_prevComp1 = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_CK]); //&cellCompParamsPtr->prevCellState->soma.Calcium_k;
	chPrms_prevComp2 = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_CL]); //&cellCompParamsPtr->prevCellState->soma.Calcium_l;
	chPrms_newComp1 = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_CK]); //&cellCompParamsPtr->newCellState->soma.Calcium_k;
	chPrms_newComp2 = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_CL]); //&cellCompParamsPtr->newCellState->soma.Calcium_l;
																		 //Compute
	SomaCalcium(chPrms_v, chPrms_prevComp1, chPrms_prevComp2, chPrms_newComp1, chPrms_newComp2);

	//Prepare pointers to inputs/outputs
	chPrms_v = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_V]); //&cellCompParamsPtr->prevCellState->soma.V_soma;
	chPrms_prevComp1 = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_SM]); //&cellCompParamsPtr->prevCellState->soma.Sodium_m;
	chPrms_prevComp2 = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_SH]); //&cellCompParamsPtr->prevCellState->soma.Sodium_h;
	chPrms_newComp1 = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_SM]); //&cellCompParamsPtr->newCellState->soma.Sodium_m;
	chPrms_newComp2 = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_SH]); //&cellCompParamsPtr->newCellState->soma.Sodium_h;
																		 //Compute
	SomaSodium(chPrms_v, chPrms_prevComp1, chPrms_prevComp2, chPrms_newComp1, chPrms_newComp2);

	//Prepare pointers to inputs/outputs
	chPrms_v = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_V]); //&cellCompParamsPtr->prevCellState->soma.V_soma;
	chPrms_prevComp1 = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_PN]); //&cellCompParamsPtr->prevCellState->soma.Potassium_n;
	chPrms_prevComp2 = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_PP]); //&cellCompParamsPtr->prevCellState->soma.Potassium_p;
	chPrms_newComp1 = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_PN]); //&cellCompParamsPtr->newCellState->soma.Potassium_n;
	chPrms_newComp2 = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_PP]); //&cellCompParamsPtr->newCellState->soma.Potassium_p;
																		 //Compute
	SomaPotassium(chPrms_v, chPrms_prevComp1, chPrms_prevComp2, chPrms_newComp1, chPrms_newComp2);

	//Prepare pointers to inputs/outputs
	chPrms_v = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_V]); //&cellCompParamsPtr->prevCellState->soma.V_soma;
	chPrms_prevComp1 = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_PXS]); //&cellCompParamsPtr->prevCellState->soma.Potassium_x_s;
	chPrms_newComp1 = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_PXS]); //&cellCompParamsPtr->newCellState->soma.Potassium_x_s;
																		  //Compute
	SomaPotassiumX(chPrms_v, chPrms_prevComp1, chPrms_newComp1);

	chComps_g_CaL = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_G]); //&cellCompParamsPtr->prevCellState->soma.g_CaL;
	chComps_vDend = &(cellCompParamsPtr[PREVSTATESTARTADD + DEND_V]); //&cellCompParamsPtr->prevCellState->dend.V_dend;
	chComps_vSoma = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_V]); //&cellCompParamsPtr->prevCellState->soma.V_soma;
	chComps_newVSoma = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_V]); //&cellCompParamsPtr->newCellState->soma.V_soma;
	chComps_vAxon = &(cellCompParamsPtr[PREVSTATESTARTADD + AXON_V]); //&cellCompParamsPtr->prevCellState->axon.V_axon;
	chComps_k = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_CK]); //&cellCompParamsPtr->newCellState->soma.Calcium_k;
	chComps_l = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_CL]); //&cellCompParamsPtr->newCellState->soma.Calcium_l;
	chComps_m = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_SM]); //&cellCompParamsPtr->newCellState->soma.Sodium_m;
	chComps_h = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_SH]); //&cellCompParamsPtr->newCellState->soma.Sodium_h;
	chComps_n = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_PN]); //&cellCompParamsPtr->newCellState->soma.Potassium_n;
	chComps_x_s = &(cellCompParamsPtr[NEXTSTATESTARTADD + SOMA_PXS]); // &cellCompParamsPtr->newCellState->soma.Potassium_x_s;
	SomaCurrVolt(chComps_g_CaL, chComps_vDend, chComps_vSoma, chComps_newVSoma, chComps_vAxon, chComps_k, chComps_l, chComps_m, chComps_h, chComps_n, chComps_x_s);

	return;
}

void SomaCalcium(user_float_t *chPrms_v, user_float_t *chPrms_prevComp1, user_float_t *chPrms_prevComp2, user_float_t *chPrms_newComp1, user_float_t *chPrms_newComp2) {

	user_float_t k_inf, l_inf, tau_k, tau_l, dk_dt, dl_dt, k_local, l_local;

	//Get inputs
	user_float_t prevV_soma = *chPrms_v; //*chPrms->v;
	user_float_t prevCalcium_k = *chPrms_prevComp1; //*chPrms->prevComp1;
	user_float_t prevCalcium_l = *chPrms_prevComp2; //*chPrms->prevComp2;

	k_inf = (1 / (1 + exp(-1 * (prevV_soma + 61) / 4.2)));
	l_inf = (1 / (1 + exp((prevV_soma + 85.5) / 8.5)));
	tau_k = 1;
	tau_l = ((20 * exp((prevV_soma + 160) / 30) / (1 + exp((prevV_soma + 84) / 7.3))) + 35);
	dk_dt = (k_inf - prevCalcium_k) / tau_k;
	dl_dt = (l_inf - prevCalcium_l) / tau_l;
	k_local = DELTA * dk_dt + prevCalcium_k;
	l_local = DELTA * dl_dt + prevCalcium_l;
	//Put result
	*chPrms_newComp1 = k_local; //*chPrms->newComp1
	*chPrms_newComp2 = l_local; //*chPrms->newComp2

	return;
}

void SomaSodium(user_float_t *chPrms_v, user_float_t *chPrms_prevComp1, user_float_t *chPrms_prevComp2, user_float_t *chPrms_newComp1, user_float_t *chPrms_newComp2) {

	user_float_t m_inf, h_inf, tau_h, dh_dt, m_local, h_local;

	//Get inputs
	user_float_t prevV_soma = *chPrms_v; //*chPrms->v;
	//user_float_t prevSodium_m = *chPrms->prevComp1;
	user_float_t prevSodium_h = *chPrms_prevComp2; //*chPrms->prevComp2;

	// RAT THALAMOCORTICAL SODIUM:
	m_inf = 1 / (1 + (exp((-30 - prevV_soma) / 5.5)));
	h_inf = 1 / (1 + (exp((-70 - prevV_soma) / -5.8)));
	tau_h = 3 * exp((-40 - prevV_soma) / 33);
	dh_dt = (h_inf - prevSodium_h) / tau_h;
	m_local = m_inf;
	h_local = prevSodium_h + DELTA * dh_dt;
	//Put result
	*chPrms_newComp1 = m_local; //*chPrms->newComp1
	*chPrms_newComp2 = h_local; //*chPrms->newComp2

	return;
}

void SomaPotassium(user_float_t *chPrms_v, user_float_t *chPrms_prevComp1, user_float_t *chPrms_prevComp2, user_float_t *chPrms_newComp1, user_float_t *chPrms_newComp2) {

	user_float_t n_inf, p_inf, tau_n, tau_p, dn_dt, dp_dt, n_local, p_local;

	//Get inputs
	user_float_t prevV_soma = *chPrms_v; //*chPrms->v;
	user_float_t prevPotassium_n = *chPrms_prevComp1; //*chPrms->prevComp1;
	user_float_t prevPotassium_p = *chPrms_prevComp2; //*chPrms->prevComp2;

	// NEOCORTICAL
	n_inf = 1 / (1 + exp((-3 - prevV_soma) / 10));
	p_inf = 1 / (1 + exp((-51 - prevV_soma) / -12));
	tau_n = 5 + (47 * exp(-(-50 - prevV_soma) / 900));
	tau_p = tau_n;
	dn_dt = (n_inf - prevPotassium_n) / tau_n;
	dp_dt = (p_inf - prevPotassium_p) / tau_p;
	n_local = DELTA * dn_dt + prevPotassium_n;
	p_local = DELTA * dp_dt + prevPotassium_p;
	//Put result
	*chPrms_newComp1 = n_local; //*chPrms->newComp1
	*chPrms_newComp2 = p_local; //*chPrms->newComp2

	return;
}

void SomaPotassiumX(user_float_t *chPrms_v, user_float_t *chPrms_prevComp1, user_float_t *chPrms_newComp1) {

	user_float_t alpha_x_s, beta_x_s, x_inf_s, tau_x_s, dx_dt_s, x_s_local;

	//Get inputs
	user_float_t prevV_soma = *chPrms_v; //*chPrms->v;
	user_float_t prevPotassium_x_s = *chPrms_prevComp1; //*chPrms->prevComp1;

	// Voltage-dependent (fast) potassium
	alpha_x_s = 0.13 * (prevV_soma + 25) / (1 - exp(-(prevV_soma + 25) / 10));
	beta_x_s = 1.69 * exp(-0.0125 * (prevV_soma + 35));
	x_inf_s = alpha_x_s / (alpha_x_s + beta_x_s);
	tau_x_s = 1 / (alpha_x_s + beta_x_s);
	dx_dt_s = (x_inf_s - prevPotassium_x_s) / tau_x_s;
	x_s_local = 0.05 * dx_dt_s + prevPotassium_x_s;
	//Put result
	*chPrms_newComp1 = x_s_local; //*chPrms->newComp1

	return;
}
void SomaCurrVolt(user_float_t *chComps_g_CaL, user_float_t *chComps_vDend, user_float_t *chComps_vSoma, user_float_t *chComps_newVSoma, user_float_t *chComps_vAxon, user_float_t *chComps_k, user_float_t *chComps_l, user_float_t *chComps_m, user_float_t *chComps_h, user_float_t *chComps_n, user_float_t *chComps_x_s) {

	//Local variables
	user_float_t I_ds, I_CaL, I_Na_s, I_ls, I_Kdr_s, I_K_s, I_as, dVs_dt;

	//Get inputs
	user_float_t g_CaL = *chComps_g_CaL; //*chComps->g_CaL;
	user_float_t prevV_dend = *chComps_vDend; //*chComps->vDend;
	user_float_t prevV_soma = *chComps_vSoma; //*chComps->vSoma;
	user_float_t prevV_axon = *chComps_vAxon; //*chComps->vAxon;
	user_float_t k = *chComps_k; //*chComps->k;
	user_float_t l = *chComps_l; //*chComps->l;
	user_float_t m = *chComps_m; //*chComps->m;
	user_float_t h = *chComps_h; //*chComps->h;
	user_float_t n = *chComps_n; //*chComps->n;
	user_float_t x_s = *chComps_x_s; //*chComps->x_s;

	// SOMATIC CURRENTS

	// Dendrite-soma interaction current I_ds
	I_ds = (G_INT / P1) * (prevV_soma - prevV_dend);
	// Inward low-threshold Ca current I_CaL
	I_CaL = g_CaL * k * k * k * l * (prevV_soma - V_CA); //k^3
	// Inward Na current I_Na_s
	I_Na_s = G_NA_S * m * m * m * h * (prevV_soma - V_NA);
	// Leakage current I_ls
	I_ls = G_LS * (prevV_soma - V_L);
	// Outward delayed potassium current I_Kdr
	I_Kdr_s = G_KDR_S * n * n * n * n * (prevV_soma - V_K); // SCHWEIGHOFER
	// I_K_s
	I_K_s = G_K_S * pow(x_s, 4) * (prevV_soma - V_K);
	// Axon-soma interaction current I_as
	I_as = (G_INT / (1 - P2)) * (prevV_soma - prevV_axon);

	dVs_dt = (-(I_CaL + I_ds + I_as + I_Na_s + I_ls + I_Kdr_s + I_K_s)) / C_M;
	*chComps_newVSoma = DELTA * dVs_dt + prevV_soma; // *chComps->newVSoma

	return;
}

void CompAxon(user_float_t *cellCompParamsPtr) {

	user_float_t *chPrms_v;
	user_float_t *chPrms_prevComp1;// *chPrms_prevComp2;
	user_float_t *chPrms_newComp1, *chPrms_newComp2;
	user_float_t *chComps_vSoma;
	user_float_t *chComps_vAxon;
	user_float_t *chComps_m_a, *chComps_h_a, *chComps_x_a;
	user_float_t *chComps_newVAxon;

	// update somatic components
	// SCHWEIGHOFER:

	//Prepare pointers to inputs/outputs
	chPrms_v = &(cellCompParamsPtr[PREVSTATESTARTADD + AXON_V]);//prevCellState->axon.V_axon;
	chPrms_prevComp1 = &(cellCompParamsPtr[PREVSTATESTARTADD + AXON_SH]);//prevCellState->axon.Sodium_h_a;
	chPrms_newComp1 = &(cellCompParamsPtr[NEXTSTATESTARTADD + AXON_SH]);//&cellCompParamsPtr->newCellState->axon.Sodium_h_a;
	chPrms_newComp2 = &(cellCompParamsPtr[NEXTSTATESTARTADD + AXON_SM]);//&cellCompParamsPtr->newCellState->axon.Sodium_m_a;
																		//Compute
	AxonSodium(chPrms_v, chPrms_prevComp1, chPrms_newComp1, chPrms_newComp2);

	//Prepare pointers to inputs/outputs
	chPrms_v = &(cellCompParamsPtr[PREVSTATESTARTADD + AXON_V]);//&cellCompParamsPtr->prevCellState->axon.V_axon;
	chPrms_prevComp1 = &(cellCompParamsPtr[PREVSTATESTARTADD + AXON_P]);//&cellCompParamsPtr->prevCellState->axon.Potassium_x_a;
	chPrms_newComp1 = &(cellCompParamsPtr[NEXTSTATESTARTADD + AXON_P]);//&cellCompParamsPtr->newCellState->axon.Potassium_x_a;
																	   //Compute
	AxonPotassium(chPrms_v, chPrms_prevComp1, chPrms_newComp1);

	//Get inputs
	chComps_vSoma = &(cellCompParamsPtr[PREVSTATESTARTADD + SOMA_V]);//&cellCompParamsPtr->prevCellState->soma.V_soma;
	chComps_vAxon = &(cellCompParamsPtr[PREVSTATESTARTADD + AXON_V]);//&cellCompParamsPtr->prevCellState->axon.V_axon;
	chComps_newVAxon = &(cellCompParamsPtr[NEXTSTATESTARTADD + AXON_V]);//&cellCompParamsPtr->newCellState->axon.V_axon;
	chComps_m_a = &(cellCompParamsPtr[NEXTSTATESTARTADD + AXON_SM]);//&cellCompParamsPtr->newCellState->axon.Sodium_m_a;
	chComps_h_a = &(cellCompParamsPtr[NEXTSTATESTARTADD + AXON_SH]);//&cellCompParamsPtr->newCellState->axon.Sodium_h_a;
	chComps_x_a = &(cellCompParamsPtr[NEXTSTATESTARTADD + AXON_P]);//&cellCompParamsPtr->newCellState->axon.Potassium_x_a;
	AxonCurrVolt(chComps_vSoma, chComps_vAxon, chComps_newVAxon, chComps_m_a, chComps_h_a, chComps_x_a);

	return;
}

void AxonSodium(user_float_t *chPrms_v, user_float_t *chPrms_prevComp1, user_float_t *chPrms_newComp1, user_float_t *chPrms_newComp2) {

	user_float_t m_inf_a, h_inf_a, tau_h_a, dh_dt_a, m_a_local, h_a_local;

	//Get inputs
	user_float_t prevV_axon = *chPrms_v; //*chPrms->v;
	user_float_t prevSodium_h_a = *chPrms_prevComp1; //*chPrms->prevComp1;

	// Update axonal Na components
	// NOTE: current has shortened inactivation to account for high
	// firing frequencies in axon hillock
	m_inf_a = 1 / (1 + (exp((-30 - prevV_axon) / 5.5)));
	h_inf_a = 1 / (1 + (exp((-60 - prevV_axon) / (-5.8))));
	tau_h_a = 1.5 * exp((-40 - prevV_axon) / 33);
	dh_dt_a = (h_inf_a - prevSodium_h_a) / tau_h_a;
	m_a_local = m_inf_a;
	h_a_local = prevSodium_h_a + DELTA * dh_dt_a;
	//Put result
	*chPrms_newComp1 = h_a_local; //*chPrms->newComp1
	*chPrms_newComp2 = m_a_local; //*chPrms->newComp2

	return;
}

void AxonPotassium(user_float_t *chPrms_v, user_float_t *chPrms_prevComp1, user_float_t *chPrms_newComp1) {

	user_float_t alpha_x_a, beta_x_a, x_inf_a, tau_x_a, dx_dt_a, x_a_local;

	//Get inputs
	user_float_t prevV_axon = *chPrms_v; //*chPrms->v;
	user_float_t prevPotassium_x_a = *chPrms_prevComp1; //*chPrms->prevComp1;

	// D'ANGELO 2001 -- Voltage-dependent potassium
	alpha_x_a = 0.13 * (prevV_axon + 25) / (1 - exp(-(prevV_axon + 25) / 10));
	beta_x_a = 1.69 * exp(-0.0125 * (prevV_axon + 35));
	x_inf_a = alpha_x_a / (alpha_x_a + beta_x_a);
	tau_x_a = 1 / (alpha_x_a + beta_x_a);
	dx_dt_a = (x_inf_a - prevPotassium_x_a) / tau_x_a;
	x_a_local = 0.05 * dx_dt_a + prevPotassium_x_a;
	//Put result
	*chPrms_newComp1 = x_a_local; //*chPrms->newComp1

	return;
}

void AxonCurrVolt(user_float_t *chComps_vSoma, user_float_t *chComps_vAxon, user_float_t *chComps_newVAxon, user_float_t *chComps_m_a, user_float_t *chComps_h_a, user_float_t *chComps_x_a) {

	//Local variable
	user_float_t I_Na_a, I_la, I_sa, I_K_a, dVa_dt;

	//Get inputs
	user_float_t prevV_soma = *chComps_vSoma; //*chComps->vSoma;
	user_float_t prevV_axon = *chComps_vAxon; //*chComps->vAxon;
	user_float_t m_a = *chComps_m_a; //*chComps->m_a;
	user_float_t h_a = *chComps_h_a; //*chComps->h_a;
	user_float_t x_a = *chComps_x_a; //*chComps->x_a;

	// AXONAL CURRENTS
	// Sodium
	I_Na_a = G_NA_A  * m_a * m_a * m_a * h_a * (prevV_axon - V_NA);
	// Leak
	I_la = G_LA    * (prevV_axon - V_L);
	// Soma-axon interaction current I_sa
	I_sa = (G_INT / P2) * (prevV_axon - prevV_soma);
	// Potassium (transient)
	//I_K_a   = G_K_A * pow(x_a, 4) * (prevV_axon - V_K);
	I_K_a = G_K_A * x_a * x_a * x_a * x_a * (prevV_axon - V_K);
	dVa_dt = (-(I_K_a + I_sa + I_la + I_Na_a)) / C_M;
	*chComps_newVAxon = DELTA * dVa_dt + prevV_axon; //*chComps->newVAxon
	return;
}

inline int dev_fetch(int j, int k) {
	return (j*IO_NETWORK_DIM1*PARAM_SIZE + k*PARAM_SIZE);
}

inline int dev_fetch_vdend(int p, int q) {
	return (p*IO_NETWORK_DIM1 + q);
}

inline void put_double(image2d_t t, int x, int y, user_float_t val) {
	//double2 d2 = ;
	//d2.x = val;
	//d2.y = 0;
	write_imageui(t, (int2)(x, y), as_uint4((double2)(val, 0.0)));
}
inline double fetch_double(image2d_t t, sampler_t sampler, int x, int y) {
	return as_double2(read_imageui(t, sampler, (int2)(x, y))).x;
	//return d2.x;
	//return 0;
}
/**
Input: cellCompParamsPtr, cellStatePtr, iApp ,i
cellCompParamsPtr: Array of struct which stores values of neighbours for each cell.
cellStatePtr: Array with values for each cell.
iApp: Extenal input of the dendrite
i: Current simulation step

Retreive the external input of the dedrite
and update the previous and new state of the current cell.
Then Compute the new variables of the current cell with ComputeOneCell.
**/
__kernel void compute_kernel(global user_float_t *cellStatePtr, global user_float_t *cellVDendPtr, const user_float_t iApp) {

	int j, k, n, p, q, e;

	k = get_global_id(0);
	j = get_global_id(1);

	user_float_t d_cellCompParams[LOCAL_PARAM_SIZE];

	d_cellCompParams[0] = iApp;

	//get neighbor v_dend
	n = 0;
	for (p = j - 1; p <= j + 1; p++) {
		for (q = k - 1; q <= k + 1; q++) {
			//cellStatePtr[dev_fetch(j, k) + (n++)] = fetch_double(cellVDendPtr, sampler, p, q);
			//if (p == j && q == k) n = n - 1;
			if (((p != j) || (q != k)) && ((p >= 0) && (q >= 0)) && ((p < IO_NETWORK_DIM1) && (q < IO_NETWORK_DIM2))) {
				//printf("k,j : %d, %d\ndev_fetch(j, k): %d\ndev_fetch_vdend(p, q): %d\nn: %d\nvdend: %lf\n", k, j, dev_fetch(j, k), dev_fetch_vdend(p, q), n, cellvdendptr[dev_fetch_vdend(p, q)]);
				//cellStatePtr[dev_fetch(j, k) + n] = cellVDendPtr[dev_fetch_vdend(p, q)];
				d_cellCompParams[VNEIGHSTARTADD + n] = cellVDendPtr[dev_fetch_vdend(p, q)];
				n++;
			} else if (p == j && q == k) {
				//	;   // do nothing, this is the cell itself
			} else {
				//printf("k,j : %d, %d\ndev_fetch(j, k): %d\ndev_fetch_vdend(j, k): %d\nn: %d\nvdend: %lf\n", k, j, dev_fetch(j, k), dev_fetch_vdend(j, k), n, cellvdendptr[dev_fetch_vdend(j, k)]);
				//cellStatePtr[dev_fetch(j, k) + n] = cellVDendPtr[dev_fetch_vdend(j, k)];
				d_cellCompParams[VNEIGHSTARTADD + n] = cellVDendPtr[dev_fetch_vdend(j, k)];
				n++;
			}
		}
	}

	/*for (int b = 0; b < PARAM_SIZE; b++) {
		printf("(state_n) %d: %lf\n", b, cellStatePtr[b]);
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int b = 0; b < LOCAL_PARAM_SIZE; b++) {
		printf("(comp_n) %d: %lf\n", b, d_cellCompParams[b]);
	}

	barrier(CLK_GLOBAL_MEM_FENCE);*/

	//Compute one by one sim step


//#pragma unroll STATEADD
//	for (e = 0; e < STATEADD; e++) {
//		d_cellCompParams[VNEIGHSTARTADD + e] = cellStatePtr[dev_fetch(j, k) + e];
//	}
#pragma unroll STATE_SIZE
	for (e = 0; e < STATE_SIZE; e++) {
		d_cellCompParams[PREVSTATESTARTADD + e] = cellStatePtr[dev_fetch(j, k) + STATEADD + e];
		d_cellCompParams[NEXTSTATESTARTADD + e] = d_cellCompParams[PREVSTATESTARTADD + e];
	}

	CompDend(d_cellCompParams);
	CompSoma(d_cellCompParams);
	CompAxon(d_cellCompParams);

	//barrier(CLK_GLOBAL_MEM_FENCE);

#pragma unroll STATE_SIZE
	for (e = 0; e < STATE_SIZE; e++) {
		cellStatePtr[dev_fetch(j, k) + STATEADD + e] = d_cellCompParams[NEXTSTATESTARTADD + e];
	}
	//put_double(cellVDendPtr, j, k, d_cellCompParams[NEXTSTATESTARTADD + DEND_V]);
	cellVDendPtr[dev_fetch_vdend(j, k)] = d_cellCompParams[NEXTSTATESTARTADD + DEND_V];
	barrier(CLK_GLOBAL_MEM_FENCE);
}

