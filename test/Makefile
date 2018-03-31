include ../Macros.mk

TOPDIR = ..

LIBS = $(patsubst %.cpp,%.o,$(wildcard ../*.cpp))
TESTS = $(patsubst %.cpp,%,$(wildcard *Test.cpp))

test: $(TESTS)
	echo '$^' | xargs -d' ' -I'{}' echo "./{}" | bash -

$(TESTS): $(LIBS) $(wildcard *Test.cpp)
	# Will have a recipe for each target in $(TESTS)
	$(CXX) -o $@ $(LIBS) $@.cpp $(LDFLAGS) $(CXXFLAGS)

.PHONY: clean

clean:
	rm -f *Test

