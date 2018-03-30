include ../macros.mk


TOPDIR = ..

## NOTE: on Ubuntu and Debian based systems you may need to add -lgfortran
  
#LIB_FLAGS = -larmadillo -library=sunperf

CXXFLAGS = $(GCC_FLAGS) $(INCLUDE) -O3

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

TESTS = $(patsubst %.cpp,%,$(wildcard *Test.cpp))
BINARIES = $(patsubst %.cpp,%,$(wildcard *Main.cpp))

LIBS_SOURCES := $(filter-out %Test.cpp,$(wildcard *.cpp))
LIBS_SOURCES := $(filter-out %Main.cpp,$(LIBS_SOURCES))
LIBS = $(patsubst %.cpp,%.o,$(LIBS_SOURCES))

all: test $(BINARIES)

$(TESTS): $(LIBS) $(wildcard *Test.cpp)
	# Will have a recipe for each target in $(TESTS)
	$(CXX) -o $@ $(LIBS) $@.cpp $(LDFLAGS) $(CXXFLAGS)

$(BINARIES): $(LIBS) $(wildcard *Main.cpp)
	$(CXX) -o $@ $@.cpp $(LIBS) $(LDFLAGS) $(CXXFLAGS)

test: $(TESTS)
	echo '$^' | xargs -d' ' -I'{}' echo "./{}" | bash -

.PHONY: clean

clean:
	rm -f $(LIBS) $(TEST_BINARY)

