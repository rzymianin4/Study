tests_run:
	for j in `seq 1 3`; do \
		make i=$$j tst ; \
		echo "————————————————————————————————" ; \
	done

tests_comp:
	for j in `seq 1 3`; do \
		make i=$$j test.x ; \
		echo "————————————————————————————————" ; \
	done

main.x: main.cpp drzewo.hpp
	g++ $< -o $@ -std=c++14

test.x: test${i}.cpp drzewo.hpp
	g++ $< -o test${i}.x -std=c++14 -lboost_unit_test_framework

tst:
	./test${i}.x --log_level=test_suite

