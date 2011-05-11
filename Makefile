TARGET     = wavefront
CXX        = g++
LINK       = g++ -fPIC
STD        = c++98
NVCC       = $(CUDA_INSTALL_PATH)/bin/nvcc
WARNINGS   = -Wall -Woverloaded-virtual -Wextra -Wpointer-arith -Wcast-qual    \
	     -Wswitch-default -Wcast-align -Wundef -Wno-empty-body             \
	     -Wreturn-type -Wformat -W -Wtrigraphs -Wno-unused-function        \
	     -Wmultichar -Wparentheses -Wchar-subscripts
INCLUDES   = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_DIR)/C/common/inc     \
	     -I$(CULA_DIR)/include -Iinclude -I$(CUDA_SDK_DIR)/shared/inc
CXXFLAGS   = $(WARNINGS) $(INCLUDES) -std=$(STD) -fopenmp
NVCCFLAGS  = $(INCLUDES) -arch sm_12
LIBS       = -lcudart -lcublas -lcula -lgomp -lshrutil_x86_64
LDFLAGS    = -L$(CUDA_INSTALL_PATH)/lib64 -L$(CUDA_SDK_DIR)/C/lib              \
	     -L$(CUDA_SDK_DIR)/C/common/lib/linux -L$(CULA_DIR)/lib64     \
	     $(LIBS) -L$(CUDA_SDK_DIR)/shared/lib

.SUFFIXES : .cu


CPPSRC = src/main.cpp \
	 src/setup.cpp \
	 src/util.cpp

CUSRC  = cu/fouriercoeff.cu \
	 cu/wavevec.cu \
	 cu/full.cu \
	 cu/secular.cu \
	 cu/inveps.cu \
	 cu/partialt.cu \
	 cu/partialr.cu


CPPOBJS = $(patsubst %.cpp,%.o,$(CPPSRC))
CUOBJS  = $(patsubst %.cu,%.o,$(CUSRC))
PTXOBJS = $(patsubst %.cu,%.o.ptx,$(CUSRC))


%.o : %.cpp
	@echo '\033[1;32m'[CC] $< '\033[1;m'
	@$(CXX) $(CXXFLAGS) -o $@ -c $<

%.o : %.cu
	@echo '\033[1;34m'[CC] $< '\033[1;m'
	@$(NVCC) $(NVCCFLAGS) -o $@ -c $<
#	@$(NVCC) $(NVCCFLAGS) -o $@.ptx -ptx $<

$(TARGET): $(CPPOBJS) $(CUOBJS)
	@$(LINK) -o $(TARGET) $(CPPOBJS) $(CUOBJS) $(LDFLAGS)
	@echo '\033[1;33m'[LD] $(TARGET) '\033[1;m'

clean:
	rm -f $(CUOBJS)
	rm -f $(CPPOBJS)
	rm -f $(PTXOBJS)
	rm -f $(TARGET)

dist:
	git archive HEAD | gzip > $(TARGETNAME).tar.gz
