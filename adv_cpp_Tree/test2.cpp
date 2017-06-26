#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE iteratorTests
#include <boost/test/unit_test.hpp>
#include <list>
#include <algorithm>
#include "drzewo.hpp"

BOOST_AUTO_TEST_CASE(rangeBasedLoop) {
	Drzewo<int> tree;
	tree.insert(0, tree.end(), 0);
	tree.insert(2, tree.root(), 0);
	auto it = tree.insert(1, tree.root(), 0);
	tree.insert(3, tree.root(), 2);
	tree.insert(4, it, 0);
	tree.insert(5, it, 1);
	std::list<int> lst;
	for(auto it: tree) {
		lst.push_back(it); }
	lst.sort();
	int i=0;
	for(auto it: lst) {
		BOOST_REQUIRE(it == i++); } }

BOOST_AUTO_TEST_CASE(constIteratorLoop) {
	Drzewo<int> tree;
	tree.insert(0, tree.end(), 0);
	tree.insert(2, tree.root(), 0);
	auto it = tree.insert(1, tree.root(), 0);
	tree.insert(3, tree.root(), 2);
	tree.insert(4, it, 0);
	tree.insert(5, it, 1);
	std::list<int> lst;
	std::for_each(tree.cbegin(), tree.cend(), [&](auto it){ lst.push_back(it); });
	lst.sort();
	int i=0;
	for(auto it: lst) {
		BOOST_REQUIRE(it == i++); } }

BOOST_AUTO_TEST_CASE(reverseIteratorLoop) {
	Drzewo<int> tree;
	tree.insert(0, tree.end(), 0);
	tree.insert(2, tree.root(), 0);
	auto it = tree.insert(1, tree.root(), 0);
	tree.insert(3, tree.root(), 2);
	tree.insert(4, it, 0);
	tree.insert(5, it, 1);
	std::list<int> lst;
	std::for_each(tree.begin(), tree.end(), [&](auto it){ lst.push_back(it); });
	std::list<int> rlst;
	std::for_each(tree.rbegin(), tree.rend(), [&](auto it){ rlst.push_back(it); });
	rlst.reverse();
	auto rit=rlst.begin();
	for(auto it: lst) {
		BOOST_REQUIRE(it == *rit);
		++rit; } }

BOOST_AUTO_TEST_CASE(constReverseIteratorLoop) {
	Drzewo<int> tree;
	tree.insert(0, tree.end(), 0);
	tree.insert(2, tree.root(), 0);
	auto it = tree.insert(1, tree.root(), 0);
	tree.insert(3, tree.root(), 2);
	tree.insert(4, it, 0);
	tree.insert(5, it, 1);
	std::list<int> lst;
	std::for_each(tree.cbegin(), tree.cend(), [&](auto it){ lst.push_back(it); });
	std::list<int> rlst;
	std::for_each(tree.crbegin(), tree.crend(), [&](auto it){ rlst.push_back(it); });
	rlst.reverse();
	auto rit=rlst.begin();
	for(auto it: lst) {
		BOOST_REQUIRE(it == *rit);
		++rit; } }
