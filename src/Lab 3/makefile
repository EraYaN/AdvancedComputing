CC							:= g++
LD							:= g++
NVCC						:= nvcc
PROJ_BASE					:= .
BIN_DIR						:= $(PROJ_BASE)/bin
EXEC_CPU					:= $(BIN_DIR)/BrainCPU
EXEC_CUDA					:= $(BIN_DIR)/BrainCUDA
EXEC_OPENCL					:= $(BIN_DIR)/BrainOpenCL
#RUN_IMAGE					:= "image00.jpg"
#RUN_ID						:= 0
CPPSTD						:= c++11

MODULES_CPU					:= ACSShared BrainCPU
SRC_DIR_CPU					:= $(addprefix ./,$(MODULES_CPU))
BUILD_DIR_CPU				:= $(addprefix out/,$(MODULES_CPU))

SRC_CPU					       := $(foreach sdir,$(SRC_DIR_CPU),$(wildcard $(sdir)/*.cpp))
OBJ_CPU						:= $(patsubst ./%.cpp,out/%.o,$(SRC_CPU))

MODULES_CUDA				:= ACSShared BrainCUDA
SRC_DIR_CUDA				   := $(addprefix ./,$(MODULES_CUDA))
BUILD_DIR_CUDA				:= $(addprefix out/,$(MODULES_CUDA))

SRC_CUDA_CPP				:= $(foreach sdir,$(SRC_DIR_CUDA),$(wildcard $(sdir)/*.cpp))
SRC_CUDA_CU					   := $(foreach sdir,$(SRC_DIR_CUDA),$(wildcard $(sdir)/*.cu))
SRC_CUDA 					:= $(SRC_CUDA_CPP) $(SRC_CUDA_CU)
OBJ_CUDA						:= $(patsubst ./%.cpp,out/%.o,$(SRC_CUDA_CPP)) $(patsubst ./%.cu,out/%.o,$(SRC_CUDA_CU))

MODULES_OPENCL					:= ACSShared BrainOpenCL
SRC_DIR_OPENCL					:= $(addprefix ./,$(MODULES_OPENCL))
BUILD_DIR_OPENCL				:= $(addprefix out/,$(MODULES_OPENCL))

SRC_OPENCL					       := $(foreach sdir,$(SRC_DIR_OPENCL),$(wildcard $(sdir)/*.cpp))
OBJ_OPENCL						:= $(patsubst ./%.cpp,out/%.o,$(SRC_OPENCL))


BUILDDIRS					:= $(sort $(BUILD_DIR_CPU) $(BUILD_DIR_CUDA) $(BUILD_DIR_OPENCL))

MODULES_INCLUDES_CPU			:= $(addprefix -I,$(MODULES_CPU))
MODULES_INCLUDES_CUDA			:= $(addprefix -I,$(MODULES_CUDA))
MODULES_INCLUDES_OPENCL			:= $(addprefix -I,$(MODULES_OPENCL))

# libjpeg-dev is not on the server for whatever reason. BAD! Only the plain dynamic lib. #-I"$(PROJ_BASE)/../jpeg-9b"
INCLUDES					:=  -I"$(CUDA_INSTALL_PATH)/include" -I"$(PROJ_BASE)/../tclap-1.2.1/include" -I"$(PROJ_BASE)/../date-master" $(MODULES_INCLUDES_CPU) $(MODULES_INCLUDES_CUDA) $(MODULES_INCLUDES_OPENCL)

INCLUDES					:= $(sort $(INCLUDES))

# libjpeg-dev is not on the server for whatever reason. BAD! Only the plain dynamic lib.
LDFLAGS						:= -lm -lX11 -lpthread -lOpenCL
CUDA_LDFLAGS				:= $(LDFLAGS) -lrt -lcudart -Xlinker -rpath=$(LD_LIBRARY_PATH)

LIBS						:=
CUDA_LIBS					:= $(LIBS) -L"$(CUDA_INSTALL_PATH)/lib64"

VPATH						:= $(PROJ_BASE)

BUILDDIR					:= $(PROJ_BASE)/out

################################################################################
# CUDA code generation flags
GENCODE_SM30				:= -gencode arch=compute_30,code=sm_30
#GENCODE_SM32				:= -gencode arch=compute_32,code=sm_32
#GENCODE_SM35				:= -gencode arch=compute_35,code=sm_35
GENCODE_SM50				:= -gencode arch=compute_50,code=sm_50
GENCODE_SMXX				:= -gencode arch=compute_50,code=compute_50
GENCODE_FLAGS				?= $(GENCODE_SM30) $(GENCODE_SM32) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SMXX)

NVCCFLAGS					:= $(GENCODE_FLAGS)
COPTS						:= -DUSE_DOUBLES

ifeq ($(USE_RDTSC),1)
	COPTS += -DUSE_RDTSC
else
	COPTS += -std=$(CPPSTD)
endif

## Debug options: -E -dI -H -v

debug := 0
ifneq ($(debug), 1)
	CFLAGS		:= -O3 -g0 $(COPTS)
	NVCCFLAGS	+= -O3 $(COPTS)
else
	CFLAGS		:= -O0 -g3 -Wall $(COPTS)
	NVCCFLAGS	+= -O0 $(COPTS)
endif

.PHONY: all checkdirs clean

all: cpu cuda opencl

cpu: checkdirs $(EXEC_CPU)

cuda: checkdirs $(EXEC_CUDA)

opencl: checkdirs $(EXEC_OPENCL)

$(EXEC_CPU): $(OBJ_CPU)
	@echo "[LD] Linking BrainCPU.."
	@$(LD) $^ -o $@ $(INCLUDES) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	@echo "[LD] Linked BrainCPU."

$(EXEC_CUDA): $(OBJ_CUDA)
	@echo "[LD] Linking BrainCUDA.."
	@$(LD) $^ -o $@ $(INCLUDES) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	@echo "[LD] Linked BrainCUDA."

$(EXEC_OPENCL): $(OBJ_OPENCL)
	@echo "[LD] Linking BrainOpenCL.."
	@$(LD) $^ -o $@ $(INCLUDES) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	@echo "[LD] Linked BrainOpenCL."
	@echo "[CP] Copying OpenCL Kernels.."
	@cp -f $(PROJ_BASE)/BrainOpenCL/*.h $(BIN_DIR)
	@cp -f $(PROJ_BASE)/BrainOpenCL/*.cl $(BIN_DIR)
	@echo "[CP] Copied OpenCL Kernels."
	

$(BUILDDIR)/%.o: %.cu
	@echo "[NVCC] Compiling $@.."
	@$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@
	@echo "[NVCC] Compiled $@."

$(BUILDDIR)/%.o: %.cpp
	@echo "[CC] Compiling $@.."
	@$(CC) $(INCLUDES) $(CFLAGS) -c $< -o $@
	@echo "[CC] Compiled $@."

checkdirs: $(BIN_DIR) $(BUILD_DIR_CPU) $(BUILD_DIR_CUDA) $(BUILD_DIR_OPENCL)

$(BIN_DIR) $(sort $(BUILD_DIR_CPU) $(BUILD_DIR_CUDA) $(BUILD_DIR_OPENCL)):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR_CPU)
	rm -rf $(BUILD_DIR_CUDA)
	rm -rf $(BUILD_DIR_OPENCL)
	rm -f $(EXEC_CPU) *~
	rm -f $(EXEC_CUDA) *~
	rm -f $(EXEC_OPENCL) *~


run: all
	$(EXEC_CPU)
	$(EXEC_CUDA)
	$(EXEC_OPENCL)

runserver: schedule
 
schedule: schedule_cpu schedule_cuda schedule_opencl
	@echo "Scheduling all benchmarks."

schedule_cpu: cpu
	@echo "Scheduling for $(EXEC_CPU) with ID: $(RUN_ID)_CPU; Profiling: $(PROFILING); Arguments: $(BENCH_ARGUMENTS)"
	@sh schedule-script.sh $(EXEC_CPU) $(RUN_ID)_CPU $(PROFILING) $(BENCH_ARGUMENTS)
	
schedule_cuda: cuda
	@echo "Scheduling for $(EXEC_CUDA) with ID: $(RUN_ID)_CUDA; Profiling: $(PROFILING); Arguments: $(BENCH_ARGUMENTS)"
	@sh schedule-script.sh $(EXEC_CUDA) $(RUN_ID)_CUDA $(PROFILING) $(BENCH_ARGUMENTS)
	
schedule_opencl: opencl
	@echo "Scheduling for $(EXEC_OPENCL) with ID: $(RUN_ID)_OPENCL; Profiling: $(PROFILING); Arguments: $(BENCH_ARGUMENTS)"
	@sh schedule-script.sh $(EXEC_OPENCL) $(RUN_ID)_OPENCL $(PROFILING) $(BENCH_ARGUMENTS)

#RUNSERVER_REQ_RUN_SETTINGS=True
#RUNSERVER_DEPS=all
#-include $(ACS_SHARED_PATH)/runserver.mk
