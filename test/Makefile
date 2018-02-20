CXX=g++

LIB_FLAGS = -lpthread -lfolly -lgflags -lglog -lgtest -larmadillo
## NOTE: on Ubuntu and Debian based systems you may need to add -lgfortran
  
#LIB_FLAGS = -larmadillo -library=sunperf

CC_FLAGS = -std=c++14 -g -Wl,--no-as-needed

OPT = -O0

TOPDIR = ..

INCLUDE = -I$(TOPDIR)
## As the Armadillo library uses recursive templates, compilation times depend on the level of optimisation:
##
## -O0: quick compilation, but the resulting program will be slow
## -O1: good trade-off between compilation time and execution speed
## -O2: produces programs which have almost all possible speedups, but compilation takes longer
## -O3: enables auto vectorisation when using gcc


#EXTRA_OPT = -fwhole-program
## Uncomment the above line if you're compiling all source files into one program in a single hit


#DEBUG = -DARMA_EXTRA_DEBUG
## Uncomment the above line to enable low-level debugging.
## Lots of debugging information will be printed when a compiled program is run.
## Please enable this option when reporting bugs.


#FINAL = -DARMA_NO_DEBUG
## Uncomment the above line to disable Armadillo's checks.
## Not recommended unless your code has been first thoroughly tested!

CXXFLAGS = $(DEBUG) $(FINAL) $(OPT) $(EXTRA_OPT) $(INCLUDE)

all: matrixTestRun registryTestRun dataTestRun

matrixTest: matrixTest.cpp
	$(CXX) $(CC_FLAGS) $(CXXFLAGS)  -o $@ $< $(LIB_FLAGS)

registryTest: RegistryTest.cpp
	$(CXX) $(CC_FLAGS) $(CXXFLAGS)  -o $@ $< $(LIB_FLAGS)

dataTest: DataTest.cpp
	$(CXX) $(CC_FLAGS) $(CXXFLAGS)  -o $@ $< $(LIB_FLAGS)

matrixTestRun: matrixTest
	./$<

registryTestRun: registryTest
	./$<

dataTestRun: dataTest
	./$<

.PHONY: clean matrixTestRun registryTestRun dataTestRun

clean:
	rm -f matrixTest registryTest dataTest

