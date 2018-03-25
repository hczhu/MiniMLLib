CXX=g++

TOPDIR = ..
INCLUDE = -I$(TOPDIR)

LDFLAGS = -lfolly -lgtest -lgflags -lglog -larmadillo -lpthread $(CPP_LIBS)

## NOTE: on Ubuntu and Debian based systems you may need to add -lgfortran
  
#LIB_FLAGS = -larmadillo -library=sunperf

CXXFLAGS = $(GCC_FLAGS) $(INCLUDE)

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

TEST_CPP = PrototypesTest.cpp
TEST_BINARY = $(patsubst %.cpp,%,$(TEST_CPP))
LIBS = $(patsubst %.cpp,%.o,$(wildcard *.cpp))

$(TEST_BINARY) : $(LIBS)
	$(CXX) -o $@ $^ $(LDFLAGS)

test: $(TEST_BINARY)
	./$(TEST_BINARY)

.PHONY: clean

clean:
	rm -f $(LIBS) $(TEST_BINARY)

