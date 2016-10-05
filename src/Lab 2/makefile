CC		:= g++
LD		:= g++
NVCC		:= nvcc
PROJ_BASE	:= .
BIN_DIR		:= $(PROJ_BASE)/bin
EXEC_MAIN		:= $(BIN_DIR)/benchmark
EXEC_COMPARE		:= $(BIN_DIR)/compare
IMAGE		:=$(PROJ_BASE)/images/image15.jpg
CPPSTD		:= c++11

MODULES   := acs-shared benchmark benchmark/cuda benchmark/cpu
SRC_DIR   := $(addprefix ./,$(MODULES))
BUILD_DIR := $(addprefix out/,$(MODULES))

SRC_CPP       := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))
SRC_CU       := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cu))
SRC			 := $(SRC_CPP) $(SRC_CU)
OBJ       := $(patsubst ./%.cpp,out/%.o,$(SRC_CPP)) $(patsubst ./%.cu,out/%.o,$(SRC_CU))

MODULES_COMPARE   := compare
SRC_DIR_COMPARE   := $(addprefix ./,$(MODULES_COMPARE))
BUILD_DIR_COMPARE := $(addprefix out/,$(MODULES_COMPARE))

SRC_COMPARE       := $(foreach sdir,$(SRC_DIR_COMPARE),$(wildcard $(sdir)/*.cpp))
OBJ_COMPARE       := $(patsubst ./%.cpp,out/%.o,$(SRC_COMPARE))

MODULES_INCLUDES := $(addprefix -I,$(MODULES))
MODULES_COMPARE_INCLUDES := $(addprefix -I,$(MODULES_COMPARE))

INCLUDES	:=  -I"$(CUDA_INSTALL_PATH)/include" -I"$(PROJ_BASE)/../tclap-1.2.1/include" -I"$(PROJ_BASE)/../date-master"  -I"$(PROJ_BASE)/../jpeg-9b" -I"$(PROJ_BASE)/../CImg-1.7.7"  $(MODULES_INCLUDES) $(MODULES_COMPARE_INCLUDES)

LDFLAGS         := -lm -lX11 -lpthread -lcimg -ljpeg
CUDA_LDFLAGS	:= $(LDFLAGS) -lrt -lcudart -Xlinker -rpath=$(LD_LIBRARY_PATH)

LIBS		:=
CUDA_LIBS	:= $(LIBS) -L"$(CUDA_INSTALL_PATH)/lib64"

VPATH = $(PROJ_BASE)

BUILDDIR = $(PROJ_BASE)/out

vpath %.cpp $(SRC_DIR)

define make-goal
$1/%.o: %.cpp
    $(CC) $(INCLUDES) -c $$< -o $$@
endef

.PHONY: all checkdirs clean

################################################################################
# CUDA code generation flags
#GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
#GENCODE_SM32    := -gencode arch=compute_32,code=sm_32
#GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SMXX    := -gencode arch=compute_50,code=compute_50
GENCODE_FLAGS   ?= $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30) \
                $(GENCODE_SM32) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SMXX)

NVCCFLAGS	:= $(GENCODE_FLAGS)

ifeq ($(USE_RDTSC),1)
	COPTS := -DUSE_RDTSC
else
    COPTS := -std=$(CPPSTD)
endif

debug		:= 0
ifneq ($(debug), 1)
	CFLAGS		:= -O3 -g0 $(COPTS)
	NVCCFLAGS	+= -O3 -Xcompiler "$(COPTS)"
else
	CFLAGS		:= -O0 -g3 -Wall $(COPTS)
	NVCCFLAGS	+= -O0 -Xcompiler "$(COPTS)"
endif

.PHONY: all checkdirs clean

all: main comp

main: checkdirs $(EXEC_MAIN)

compare: checkdirs $(EXEC_COMPARE)

$(EXEC_MAIN): $(OBJ)
	$(LD) $^ -o $@ $(INCLUDES) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)

$(EXEC_COMPARE): $(OBJ_COMPARE)
	$(LD) $^ -o $@ $(INCLUDES) $(CFLAGS)

$(BUILDDIR)/%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $<

$(BUILDDIR)/%.o: %.cpp
	$(CC) $(INCLUDES) $(CFLAGS) -c $< -o $@

$(BUILDDIR_COMPARE)/%.o: %.cpp
	$(CC) $(INCLUDES) $(CFLAGS) -c $< -o $@

checkdirs: $(BIN_DIR) $(BUILD_DIR) $(BUILD_DIR_COMPARE)

$(BIN_DIR) $(BUILD_DIR) $(BUILD_DIR_COMPARE):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(BUILD_DIR_COMPARE)
	rm -f $(EXEC_MAIN) *~
	rm -f $(EXEC_COMPARE) *~

run: all
	$(EXEC_MAIN) $(IMAGE)

RUNSERVER_REQ_RUN_SETTINGS=True
RUNSERVER_DEPS=all
-include $(ACS_SHARED_PATH)/runserver.mk


