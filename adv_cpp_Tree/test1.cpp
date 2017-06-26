#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE BasictTests
#include <boost/test/unit_test.hpp>
#include "drzewo.hpp"

BOOST_AUTO_TEST_CASE(emptyTree) {
	Drzewo<int> tree;
	BOOST_REQUIRE(tree.empty());
	BOOST_REQUIRE(tree.size() == 0); }

BOOST_AUTO_TEST_CASE(oneElementConstructor) {
	int a = 6;
	Drzewo<int> tree(a);
	Drzewo<int> bush(5); 
	BOOST_REQUIRE(!tree.empty());
	BOOST_REQUIRE(tree.size() == 1);
	BOOST_REQUIRE(!bush.empty());
	BOOST_REQUIRE(bush.size() == 1); }

BOOST_AUTO_TEST_CASE(insertion) {
	Drzewo<int> tree;
	int s = 0;
	tree.insert(s, tree.end(), 0);
	tree.insert(1, tree.root(), 0);
	tree.insert(2, tree.root(), 1);
	tree.insert(3, tree.root(), 2);
	BOOST_REQUIRE(!tree.empty());
	BOOST_REQUIRE(tree.size() == 4); }

BOOST_AUTO_TEST_CASE(erase) {
	Drzewo<int> tree;
	int s = 0;
	tree.insert(s, tree.end(), 0);
	tree.insert(1, tree.root(), 0);
	tree.insert(2, tree.root(), 1);
	tree.insert(3, tree.root(), 2);
	auto it = tree.erase(++tree.root());
	BOOST_REQUIRE(!tree.empty());
	BOOST_REQUIRE(tree.size() == 3);
	BOOST_REQUIRE(it == tree.root()); }