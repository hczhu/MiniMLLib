include ../Macros.mk

TOPDIR = ..

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
	rm -f $(LIBS) $(TESTS) $(BINARIES)

