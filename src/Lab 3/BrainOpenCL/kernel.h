#include "variables.h"

typedef double user_float_t;

inline int dev_fetch(int j, int k);

void ComputeOneCell(user_float_t *);
void CompDend(user_float_t *cellCompParamsPtr);
void DendHCurr(user_float_t *chPrms_v, user_float_t *chPrms_prevComp1, user_float_t *chPrms_newComp1);
void DendCaCurr(user_float_t *chPrms_v, user_float_t *chPrms_prevComp1, user_float_t *chPrms_newComp1);
void DendKCurr(user_float_t *chPrms_prevComp1, user_float_t *chPrms_prevComp2, user_float_t *chPrms_newComp1);
void DendCal(user_float_t *chPrms_prevComp1, user_float_t *chPrms_prevComp2, user_float_t *chPrms_newComp1);
void DendCurrVolt(user_float_t chComps_iC, user_float_t *chComps_iApp, user_float_t *chComps_vDend, user_float_t *chComps_newVDend, user_float_t *chComps_vSoma, user_float_t *chComps_q, user_float_t *chComps_r, user_float_t *chComps_s, user_float_t *chComps_newI_CaH);
user_float_t IcNeighbors(user_float_t *neighVdend, user_float_t prevV_dend);
void CompSoma(user_float_t *cellCompParamsPtr);
void SomaCalcium(user_float_t *chPrms_v, user_float_t *chPrms_prevComp1, user_float_t *chPrms_prevComp2, user_float_t *chPrms_newComp1, user_float_t *chPrms_newComp2);
void SomaSodium(user_float_t *chPrms_v, user_float_t *chPrms_prevComp1, user_float_t *chPrms_prevComp2, user_float_t *chPrms_newComp1, user_float_t *chPrms_newComp2);
void SomaPotassium(user_float_t *chPrms_v, user_float_t *chPrms_prevComp1, user_float_t *chPrms_prevComp2, user_float_t *chPrms_newComp1, user_float_t *chPrms_newComp2);
void SomaPotassiumX(user_float_t *chPrms_v, user_float_t *chPrms_prevComp1, user_float_t *chPrms_newComp1);
void SomaCurrVolt(user_float_t *chComps_g_CaL, user_float_t *chComps_vDend, user_float_t *chComps_vSoma, user_float_t *chComps_newVSoma, user_float_t *chComps_vAxon, user_float_t *chComps_k, user_float_t *chComps_l, user_float_t *chComps_m, user_float_t *chComps_h, user_float_t *chComps_n, user_float_t *chComps_x_s);
void CompAxon(user_float_t *cellCompParamsPtr);
void AxonSodium(user_float_t *chPrms_v, user_float_t *chPrms_prevComp1, user_float_t *chPrms_newComp1, user_float_t *chPrms_newComp2);
void AxonPotassium(user_float_t *chPrms_v, user_float_t *chPrms_prevComp1, user_float_t *chPrms_newComp1);
void AxonCurrVolt(user_float_t *chComps_vSoma, user_float_t *chComps_vAxon, user_float_t *chComps_newVAxon, user_float_t *chComps_m_a, user_float_t *chComps_h_a, user_float_t *chComps_x_a);