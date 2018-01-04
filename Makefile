CXX = g++
#-std=c++0x
CXXFLAGS = -std=c++11 -Wall -Ofast -fopenmp -pthread -I/usr/local/eigen -I/usr/local/gsl/include -I/usr/local/boost/include
LDFLAGS = -L/usr/local/gsl/lib -L/usr/local/boost/lib
LDLIBS = -lgsl -lopenblas -lm
TARGET = main
SRCS = main.cc src/utils.cc src/read_file.cc src/SVRG.cc
OBJS = $(SRCS:.cc=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS) $(LDLIBS)

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS)
