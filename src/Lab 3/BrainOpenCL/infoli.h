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
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include "variables.h" 
#include "timing.h"
#include "string_format.h"
#include "exit_codes.h"
#include "opencl_helpers.h"
#include "interactive_tools.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


#ifdef DEBUG
#define DEBUG_PRINT(x) std::printf x
#else
#define DEBUG_PRINT(x) do {} while (0)
#endif

#ifdef USE_DOUBLES
typedef cl_double cl_float_t;
#else
typedef float cl_float_t;
#endif