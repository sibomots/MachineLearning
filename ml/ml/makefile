## makefile for non-windows platforms

CXX = g++

CXXFLAGS :=  -g -D_DEBUG -Iinc

EXE := prog
OBJ := main.o data.o data_handler.o knn.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

prog: $(OBJ)
	$(CXX) $^ -o $@

clean:
	rm -f $(OBJ) $(EXE)
