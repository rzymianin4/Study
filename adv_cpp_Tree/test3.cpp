#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ClearAndMoveTests
#include <boost/test/unit_test.hpp>
#include "drzewo.hpp"

BOOST_AUTO_TEST_CASE(clearAll) {
	Drzewo<int> tree;
	tree.insert(0, tree.end(), 0);
	tree.insert(2, tree.root(), 0);
	auto it = tree.insert(1, tree.root(), 0);
	tree.insert(3, tree.root(), 2);
	tree.insert(4, it, 0);
	tree.insert(5, it, 1);
	BOOST_REQUIRE(tree.size() == 6);
	tree.clear();
	BOOST_REQUIRE(tree.empty()); }
	
BOOST_AUTO_TEST_CASE(clearPartial) {
	Drzewo<int> tree;
	tree.insert(0, tree.end(), 0);
	tree.insert(2, tree.root(), 0);
	auto it = tree.insert(1, tree.root(), 0);
	tree.insert(3, tree.root(), 2);
	tree.insert(4, it, 0);
	tree.insert(5, it, 1);
	BOOST_REQUIRE(tree.size() == 6);
	tree.clear(it);
	BOOST_REQUIRE(tree.size() == 3); }

BOOST_AUTO_TEST_CASE(copyOperator) {
	Drzewo<int> bush;
	bush.insert(0, bush.end(), 0);
	bush.insert(1, bush.root(), 0);
	bush.insert(2, bush.root(), 1);

	Drzewo<int> tree;
	tree.insert(0, tree.end(), 0);
	tree.insert(2, tree.root(), 0);
	auto it = tree.insert(1, tree.root(), 0);
	tree.insert(3, tree.root(), 2);
	tree.insert(4, it, 0);
	tree.insert(5, it, 1);

	BOOST_REQUIRE(tree.size() == 6);
	BOOST_REQUIRE(bush.size() == 3);

	bush = tree;
	auto it_t = tree.begin();
	auto it_b = bush.begin();
	while(it_t != tree.end() || it_b != bush.end()) {
		BOOST_REQUIRE(*it_t == *it_b);
		++it_t;
		++it_b; }
	bush.erase(bush.rbegin());
	BOOST_REQUIRE(tree.size() == 6);
	BOOST_REQUIRE(bush.size() == 5); }
	

