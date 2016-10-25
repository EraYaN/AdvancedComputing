#pragma once
/*
 *
 * Copyright (c) 2012, Neurasmus B.V., The Netherlands,
 * web: www.neurasmus.com email: info@neurasmus.com
 *
 * Any use reproduction in whole or in parts is prohibited
 * without the written consent of the copyright owner.
 *
 * All Rights Reserved.
 *
 *
 * Author: Sebastian Isaza
 * Created: 10-04-2012
 * Modified: 06-06-2012
 *
 * Description : Top header file of the Inferior Olive model. It contains the
 * constant model conductances, the data structures that hold the cell state and
 * the function prototypes.
 *
 */
#define __STDC_WANT_LIB_EXT1__ 1
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "timing.h"
#include "user_types.h"
#include "string_format.h"
#include "interactive_tools.h"

/*** MACROS ***/
#define RAND_INIT 0 // make it zero to facilitate debugging
#define SIMTIME 500 // in ms, for when no input file is provided
//IO network size is IO_NETWORK_DIM1*IO_NETWORK_DIM2
#define IO_NETWORK_DIM1 4
#define IO_NETWORK_DIM2 4
#define IO_NETWORK_SIZE IO_NETWORK_DIM1*IO_NETWORK_DIM2

#define IAPP_MAX_CHARS 6 //2 integer, the dot, 2 decimals and the delimiter

// Cell properties
#define DELTA 0.05
//Conductance for neighbors' coupling
#define CONDUCTANCE 0.04
// Capacitance
#define C_M 1
// Somatic conductances (mS/cm2)
#define G_NA_S 150      // Na gate conductance (=90 in Schweighofer code, 70 in paper) 120 too little
#define G_KDR_S 9.0    // K delayed rectifier gate conductance (alternative value: 18)
#define G_K_S 5      // Voltage-dependent (fast) potassium
#define G_LS 0.016  // Leak conductance (0.015)
// Dendritic conductances (mS/cm2)
#define G_K_CA 35       // Potassium gate conductance (35)
#define G_CAH 4.5     // High-threshold Ca gate conductance (4.5)
#define G_LD 0.016   // Dendrite leak conductance (0.015)
#define G_H 0.125    // H current gate conductance (1.5) (0.15 in SCHWEIGHOFER 2004)
// Axon hillock conductances (mS/cm2)
#define G_NA_A 240      // Na gate conductance (according to literature: 100 to 200 times as big as somatic conductance)
#define G_NA_R 0      // Na (resurgent) gate conductance
#define G_K_A 20      // K voltage-dependent
#define G_LA 0.016  // Leak conductance
// Cell morphology
#define P1 0.25        // Cell surface ratio soma/dendrite (0.2)
#define P2 0.15      // Cell surface ratio axon(hillock)/soma (0.1)
#define G_INT 0.13       // Cell internal conductance (0.13)
// Reversal potentials
#define V_NA 55       // Na reversal potential (55)
#define V_K -75       // K reversal potential
#define V_CA 120       // Ca reversal potential (120)
#define V_H -43       // H current reversal potential
#define V_L 10       // leak current

#define DEBUG 1 // comment this if nothing needs to be printed to the console
#define WRITE_OUTPUT 1
#define EXTRA_TIMING 1

#ifdef DEBUG
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x) do {} while (0)
#endif


/*** TYPEDEFS AND STRUCTS***/

struct dend{
	user_float_t V_dend;
	user_float_t Hcurrent_q;
	user_float_t Calcium_r;
	user_float_t Potassium_s;
	user_float_t I_CaH;
	user_float_t Ca2Plus;
};

struct soma{
	user_float_t g_CaL;
	user_float_t V_soma;
	user_float_t Sodium_m;
	user_float_t Sodium_h;
	user_float_t Calcium_k;
	user_float_t Calcium_l;
	user_float_t Potassium_n;
	user_float_t Potassium_p;
	user_float_t Potassium_x_s;
};

struct axon{
	user_float_t V_axon;
	user_float_t Sodium_m_a;
	user_float_t Sodium_h_a;
	user_float_t Potassium_x_a;
};

typedef struct cellState{
	struct dend dend;
	struct soma soma;
	struct axon axon;
}cellState;

typedef struct cellCompParams{
	user_float_t iAppIn;
	user_float_t neighVdend[15];
	cellState *prevCellState;
	cellState *newCellState;
}cellCompParams;

typedef struct channelParams{
	user_float_t *v;
	user_float_t *prevComp1, *prevComp2;
	user_float_t *newComp1, *newComp2;
}channelParams;

typedef struct dendCurrVoltPrms{
	user_float_t *iApp;
	user_float_t iC;
	user_float_t *vDend;
	user_float_t *vSoma;
	user_float_t *q, *r, *s;
	user_float_t *newVDend;
	user_float_t *newI_CaH;
}dendCurrVoltPrms;

typedef struct somaCurrVoltPrms{
	user_float_t *g_CaL;
	user_float_t *vSoma;
	user_float_t *vDend;
	user_float_t *vAxon;
	user_float_t *k, *l, *m, *h, *n, *x_s;
	user_float_t *newVSoma;
}somaCurrVoltPrms;

typedef struct axonCurrVoltPrms{
	user_float_t *vSoma;
	user_float_t *vAxon;
	user_float_t *m_a, *h_a, *x_a;
	user_float_t *newVAxon;
}axonCurrVoltPrms;

/*** FUNCTION PROTOTYPES ***/
void neighbors(cellCompParams **cellCompParamsPtr, cellState ***cellStatePtr,int i, int j, int k);
void compute(cellCompParams **cellCompParamsPtr, cellState ***cellStatePtr, int iApp, int i, int j, int k);
void ComputeOneCell(cellCompParams *);

void CompDend(cellCompParams *);
void DendHCurr(struct channelParams *);
void DendCaCurr(struct channelParams *);
void DendKCurr(struct channelParams *);
void DendCal(struct channelParams *);
void DendCurrVolt(struct dendCurrVoltPrms *);
user_float_t IcNeighbors(user_float_t *, user_float_t);

void CompSoma(cellCompParams *);
void SomaCalcium(struct channelParams *);
void SomaSodium(struct channelParams *);
void SomaPotassium(struct channelParams *);
void SomaPotassiumX(struct channelParams *);
void SomaCurrVolt(struct somaCurrVoltPrms *);

void CompAxon(cellCompParams *);
void AxonSodium(channelParams *);
void AxonPotassium(channelParams *);
void AxonCurrVolt(axonCurrVoltPrms *);

//inline user_float_t min(user_float_t a, user_float_t b);

void mallocCells(cellCompParams ***cellCompParamsPtr, cellState ****cellStatePtr);
void InitState(cellState **);
void init_g_CaL(cellState ***cellStatePtr);
void random_init(cellCompParams **cellCompParamsPtr, cellState ***cellStatePtr);
