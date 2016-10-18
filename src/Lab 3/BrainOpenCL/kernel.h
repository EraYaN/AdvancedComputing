#include "variables.h"

typedef double mod_prec;
//typedef float mod_prec;

void ComputeOneCell(mod_prec *);
void CompDend(mod_prec *cellCompParamsPtr);
void DendHCurr(mod_prec *chPrms_v, mod_prec *chPrms_prevComp1, mod_prec *chPrms_newComp1);
void DendCaCurr(mod_prec *chPrms_v, mod_prec *chPrms_prevComp1, mod_prec *chPrms_newComp1);
void DendKCurr(mod_prec *chPrms_prevComp1, mod_prec *chPrms_prevComp2, mod_prec *chPrms_newComp1);
void DendCal(mod_prec *chPrms_prevComp1, mod_prec *chPrms_prevComp2, mod_prec *chPrms_newComp1);
void DendCurrVolt(mod_prec chComps_iC, mod_prec *chComps_iApp, mod_prec *chComps_vDend, mod_prec *chComps_newVDend, mod_prec *chComps_vSoma, mod_prec *chComps_q, mod_prec *chComps_r, mod_prec *chComps_s, mod_prec *chComps_newI_CaH);
mod_prec IcNeighbors(mod_prec *neighVdend, mod_prec prevV_dend);
void CompSoma(mod_prec *cellCompParamsPtr);
void SomaCalcium(mod_prec *chPrms_v, mod_prec *chPrms_prevComp1, mod_prec *chPrms_prevComp2, mod_prec *chPrms_newComp1, mod_prec *chPrms_newComp2);
void SomaSodium(mod_prec *chPrms_v, mod_prec *chPrms_prevComp1, mod_prec *chPrms_prevComp2, mod_prec *chPrms_newComp1, mod_prec *chPrms_newComp2);
void SomaPotassium(mod_prec *chPrms_v, mod_prec *chPrms_prevComp1, mod_prec *chPrms_prevComp2, mod_prec *chPrms_newComp1, mod_prec *chPrms_newComp2);
void SomaPotassiumX(mod_prec *chPrms_v, mod_prec *chPrms_prevComp1, mod_prec *chPrms_newComp1);
void SomaCurrVolt(mod_prec *chComps_g_CaL, mod_prec *chComps_vDend, mod_prec *chComps_vSoma, mod_prec *chComps_newVSoma, mod_prec *chComps_vAxon, mod_prec *chComps_k, mod_prec *chComps_l, mod_prec *chComps_m, mod_prec *chComps_h, mod_prec *chComps_n, mod_prec *chComps_x_s);
void CompAxon(mod_prec *cellCompParamsPtr);
void AxonSodium(mod_prec *chPrms_v, mod_prec *chPrms_prevComp1, mod_prec *chPrms_newComp1, mod_prec *chPrms_newComp2);
void AxonPotassium(mod_prec *chPrms_v, mod_prec *chPrms_prevComp1, mod_prec *chPrms_newComp1);
void AxonCurrVolt(mod_prec *chComps_vSoma, mod_prec *chComps_vAxon, mod_prec *chComps_newVAxon, mod_prec *chComps_m_a, mod_prec *chComps_h_a, mod_prec *chComps_x_a);