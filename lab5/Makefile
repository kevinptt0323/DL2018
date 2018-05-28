CXX = g++
CXXFLAGS = -std=c++14 -O3 -g -Wall -fmessage-length=0
OBJ = td-after td

all: $(OBJ)

$(OBJ): %:%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
clean:
	rm $(OBJ)
