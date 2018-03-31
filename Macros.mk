CXX=g++
INCLUDE = -I$(TOPDIR)
LDFLAGS = -lfolly -lgtest -lgflags -lglog -larmadillo -lpthread -ldouble-conversion
CXXFLAGS = $(GCC_FLAGS) $(INCLUDE) -O3
