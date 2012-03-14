CC=g++ -O3 -fopenmp
#CC=clang++ -O3
LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc
OBJS=Connected.o HammingHash.o PlaneHeuristics.o Simplification.o
EXTRA_OBJS=ColorMap.o
INC=-Iinclude

segment: demo/main.cpp ${OBJS} ${EXTRA_OBJS}
	${CC} ${INC} $^ ${LIBS} -o segment

$(OBJS): %.o: src/%.cpp
	${CC} ${INC} -c $< -o $@

$(EXTRA_OBJS): %.o: extra/%.cpp
	${CC} ${INC} -c $< -o $@

.PHONY : clean

clean:
	rm -f *.o segment
