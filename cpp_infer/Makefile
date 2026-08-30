CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=native -ffast-math -funroll-loops -fomit-frame-pointer -fno-rtti -fno-exceptions -Wall -Wextra -I.
LDFLAGS ?= -lzstd

TARGET = affine_infer_cli
SRCS = affine_infer.cpp main.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cpp affine_infer.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
