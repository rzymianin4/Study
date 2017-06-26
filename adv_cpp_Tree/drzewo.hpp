#include <list>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <exception>

template<typename type> class Drzewo;
template<typename type> class Iterator;
template<typename type> class ConstIterator;
template<typename type> class ReverseIterator;
template<typename type> class ConstReverseIterator;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

template<typename type> class Node {
	private:
		type _obj;
		Node<type>* _parent;
		std::list<Node<type>*> _children;
		typename std::list<Node<type>*>::iterator _iter; // iter for this node in parent's children list
	
	public:
		//Node<type>(): _obj(type()), _parent(nullptr), _children(nullptr) { }
		template<typename ref>
		Node<type>(ref&& obj, Node<type>* parent=nullptr):
			_obj(std::forward<type>(obj)), _parent(parent), _children(std::list<Node<type>*>()) { }
		
		auto get_iter() {
			if(!_parent) {
				throw std::invalid_argument("root doesn't have a parent, so does not have an index"); }
			return _iter; }

		operator type&() {
			return _obj; }
		
		friend class Iterator<type>;
		friend class Drzewo<type>;
};


/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

/**
 * Tree-shape contanair storing values of decalated types.
 * Template takes exactyly one typnename parameter.
 */
template<typename type> class Drzewo {
	private:
		Node<type>* _root;
		std::size_t _size;

		Node<type>* _copy_node(Node<type>&, Node<type>* = nullptr);
	
	public:
		typedef type value_type;
		typedef std::size_t size_type;
		typedef type& reference;
		typedef const type& const_reference;
		typedef type* pointer;
		typedef const type* const_pointer;
		typedef Iterator<type> iterator;
		typedef ConstIterator<type> const_iterator;

		/**
		 * None-argument constructor. Initializes empty tree.
		 */		
		Drzewo<type>(): _root(nullptr), _size(0) { }

		/**
		 * One-argument constructor.
		 *
		 * @param obj 	right-value or reference to object, which will become root of new tree
		 */		
		template<typename ref>
		Drzewo<type>(ref&& obj): _root(new Node<type>(std::forward<type>(obj))), _size(1) { }

		/**
		 * Copy constructor.
		 *
		 * @param tree 	const reference to tree that will be copyed
		 */		
		Drzewo<type>(Drzewo<type>& tree): _root(nullptr), _size(tree.size()) {
			_root = _copy_node(*tree._root);
			_size = tree._size; }

		/**
		 * Destructor. Clears all allocated memory.
		 */	
		~Drzewo<type>() { clear(); }

		/**
		 * Copy operator=.
		 *
		 * @param tree 	const reference to tree
		 *
		 * @return 		state of the tree after operation
		 */
		Drzewo<type>& operator=(const Drzewo<type>& tree) {
			clear();
			_root = _copy_node(*tree._root);
			_size = tree._size;
			return *this; }

		/**
		 * Move operator=.
		 *
		 * @param tree 	r-value reference to tree
		 *
		 * @return 		state of the tree after operation
		 */
		Drzewo<type>& operator=(Drzewo<type>&& tree) {
			clear();
			_root = tree._root;
			tree._root = nullptr;
			tree._size = 0;
			return *this; }

		/**
		 * Check size of container.
		 *
		 * @return 		number of lements actually stored in the container
		 */	
		std::size_t size() const {
			return _size; }

		/**
		 * Check if container is empty.
		 *
		 * @return 		<b><i>true</i></b> if conainer is emty, or <b><I>false</i></b> if it
		 * 				contains at least one element
		 */	
		bool empty() const {
			return _size==0 ? true : false; }
		
		/**
		 * Random access operator. (yes, it's a little joke)
		 *
		 * @param index 	index of an element in the tree
		 *
		 * @return 		reference to random element of the tree
		 *
		 * @throw		std::out_of_range	if <b>index</b> is greater than tree
		 */
		type& operator[](int)
			throw(std::out_of_range);
		
		/**
		 * Insert specify element as a child of given node on the given position.
		 *
		 * @param obj 		element to insert
		 * @param parent	iterator pointing to node that should be future parent
		 * @param index		specyfies position in list of childern of parent node
		 *
		 * @return		iterator to the inserted element
		 *
		 * @throw std::invalid_argument	if <b>parent</b> is </b><i>end()</i></b> and tree alredy has a root,
		 *								or <b>parent</b> points to another tree
		 * @throw std::out_of_range		if <b>index</b> is greater than number of <b>parent</b>'s children
		 */
		template<typename ref>
		Iterator<type> insert(ref&&, Iterator<type>, int)
			throw(std::invalid_argument, std::out_of_range);
		
		/**
		 * Erase specify element from he tree if it has no children.
		 *
		 * @param element	element to erase
		 *
		 * @return 		iterator to the parent of the erased element
		 *
		 * @throw std::invalid_argument	if <b>element</b> has any children,
		 *								or <b>element</b> points to another tree
		 * @throw std::out_of_range		if <b>element</b> is </b><i>end()</i></b
		 */
		Iterator<type> erase(Iterator<type> element)
			throw(std::invalid_argument, std::out_of_range);

		/**
		 * Removes given node and all nodes below. If no node is given clears all tree.
		 *
		 * @param node 	Pointer to the node to remove. Could be an Iterator.
		 */
		void clear(Node<type>* node = nullptr);

		/**
		 * Access to the root. Equal to <i><b>begin()</b></i>.
		 *
		 * @param element	element to erase
		 *
		 * @return 		iterator to the root
		 */
		Iterator<type> root() {
			return Iterator<type>(this, _root); }

		/**
		 * Access to the first element of the tree. Equal to <i><b>root()</b></i>.
		 *
		 * @param element	element to erase
		 *
		 * @return 		iterator to the first element of the tree
		 */
		Iterator<type> begin() {
			return Iterator<type>(this, _root); }
		ConstIterator<type> cbegin() {
			return ConstIterator<type>(this, _root); }

		/**
		 * Access to the last element of the tree (last right leaf).
		 *
		 * @param element	element to erase
		 *
		 * @return 		reverse iterator to the last element of the tree
		 */
		ReverseIterator<type> rbegin() {
			auto iter = ReverseIterator<type>(this, _root);
			while(!iter._node->_children.empty()) {
				iter._node = iter._node->_children.back(); }
				return iter; }
		ConstReverseIterator<type> crbegin() {
			auto iter = ConstReverseIterator<type>(this, _root);
			while(!iter._node->_children.empty()) {
				iter._node = iter._node->_children.back(); }
				return iter; }

		/**
		 * Get iterator pointing out of tree (to the past-the-last element, in fact same as rend()).
		 *
		 * @param element	element to erase
		 *
		 * @return 		iterator to the last element
		 */
		Iterator<type> end() {
			return Iterator<type>(this); }
		ConstIterator<type> cend() {
			return ConstIterator<type>(this); }

		/**
		 * Get iterator pointing out of tree (to the pre-the-first element, in fact same as end()).
		 *
		 * @param element	element to erase
		 *
		 * @return 		reverse iterator to the last element
		 */
		ReverseIterator<type> rend() {
			return ReverseIterator<type>(this); }
		ConstReverseIterator<type> crend() {
			return ConstReverseIterator<type>(this); }

		/**
		 * Get number of children of node pointing by given iterator;
		 *
		 * @param iter	iterator of const_iterator
		 *
		 * @return 		number of children of the node
		 */
		template<typename itertype>
		int getNumberOfChildren(const itertype& iter) const {
			return iter._node->_children.size(); }

		/**
		 * Get child of node pointing by given iterator, on specifyed position;
		 *
		 * @param iter	iterator of const_iterator
		 * @param index	index of required child on the list of children
		 *
		 * @return 		iterator pointing to required child
		 */
		template<typename itertype>
		auto getChild(const itertype& iter, int index) const throw(std::out_of_range) {
			return iter.getChild(index); }

		friend class Iterator<type>;
};


/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

template<typename type> class Iterator {
	protected:
		Iterator<type>() { }
		Drzewo<type>* _tree;
		Node<type>* _node;

		Iterator<type>(Drzewo<type>* tree, Node<type>* node=nullptr): _tree(tree), _node(node) { }

	public:
		Drzewo<type>* get_tree() {
			return _tree; }

		operator Node<type>*() {
			return _node; }

		bool operator==(Iterator<type> iter) {
			//cout << _node << " " << iter._node << endl;
			return _node == iter._node; }
		bool operator!=(Iterator<type> iter) {
			return _node != iter._node; }
		type& operator*() throw(std::out_of_range) {
			if(_node) { return *_node; } // implicit conversion
			else { throw std::out_of_range("iterator out of tree"); } }
		type* operator->() throw(std::out_of_range) {
			if(_node) { return &_node->_obj; } // implicit conversion
			else { throw std::out_of_range("iterator out of tree"); } }

		int getNumberOfChildren() {
			return _node->_children.size(); }
		Iterator<type> getChild(int index) throw(std::out_of_range);

		virtual Iterator<type>& operator++() throw(std::out_of_range);
		virtual Iterator<type> operator++(int) throw(std::out_of_range);
		virtual Iterator<type>& operator--() throw(std::out_of_range);
		virtual Iterator<type> operator--(int) throw(std::out_of_range);

		friend class Drzewo<type>;
};


/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

template<typename type> class ConstIterator: public virtual Iterator<type> {
	protected:
		ConstIterator<type>() { }
		ConstIterator<type>(Drzewo<type>* tree, Node<type>* node=nullptr): Iterator<type>(tree, node) { }

	public:
		ConstIterator<type>(Iterator<type> iter): Iterator<type>(iter._tree, iter._node) { }
		
		const type& operator*() throw(std::out_of_range) {
			if(this->_node) { return *this->_node; } // implicit conversion
			else { throw std::out_of_range("iterator out of tree"); } }
		const type* operator->() throw(std::out_of_range) {
			if(this->_node) { return &this->_node->_obj; } // implicit conversion
			else { throw std::out_of_range("iterator out of tree"); } }

		friend class Drzewo<type>;
};


/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

template<typename type> class ReverseIterator: public virtual Iterator<type> {
	protected:
		ReverseIterator<type>() { }
		ReverseIterator<type>(Drzewo<type>* tree, Node<type>* node=nullptr): Iterator<type>(tree, node) { }

	public:
		ReverseIterator<type>(Iterator<type> iter): Iterator<type>(iter._tree, iter._node) { }
		
		Iterator<type>& operator++() throw(std::out_of_range) {
			return Iterator<type>::operator--(); }
		Iterator<type> operator++(int) throw(std::out_of_range) {
			auto iter = *this;
			++*this;
			return iter; }
		Iterator<type>& operator--() throw(std::out_of_range) {
			return Iterator<type>::operator++(); }
		Iterator<type> operator--(int) throw(std::out_of_range) {
			auto iter = *this;
			--*this;
			return iter; }

		friend class Drzewo<type>;
};

template<typename type> class ConstReverseIterator: public ConstIterator<type>, 
													public ReverseIterator<type> {
	private:
		ConstReverseIterator<type>(Drzewo<type>* tree, Node<type>* node=nullptr):
			Iterator<type>(tree, node) { }

	public:
		ConstReverseIterator<type>(Iterator<type> iter): Iterator<type>(iter._tree, iter._node) { }

		friend class Drzewo<type>;
};

/*-------------------------------------------------------------------------------------------------------*/


/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
// DRZEWO

template<typename type>
Node<type>* Drzewo<type>::_copy_node(Node<type>& ntcp, Node<type>* parent) {
	Node<type>* n_node = new Node<type>(ntcp._obj, parent);
	for(Node<type>* iter: ntcp._children) {
		n_node->_children.push_back(_copy_node(*iter, n_node));
		n_node->_children.back()->_iter = --n_node->_children.end(); }
	return n_node; }

template<typename type>
type& Drzewo<type>::operator[](int index)
throw(std::out_of_range) {
	if(index>=_size) {
		throw std::out_of_range("index out of range"); }
	srand(time(NULL));
	index = std::rand()%_size;
	auto it = begin();
	for(int i=0; i<index; i++) {
		++it; }
	return *it; }

template<typename type>
template<typename ref>
Iterator<type> Drzewo<type>::insert(ref&& obj, Iterator<type> parent, int index)
throw(std::invalid_argument, std::out_of_range) {
	if(parent.get_tree() != this) {
		throw std::invalid_argument("given iterator points to another tree"); }
	if(parent == end()) {
		if(_size == 0) {
			_root = new Node<type>(std::forward<type>(obj));
			_size++;
			return root(); }
		else {
			throw std::invalid_argument("this tree alredy has a root"); } }
	if(index > parent._node->_children.size()) {
		throw std::out_of_range("given node doesn't have as many children"); }
	auto it = parent._node->_children.begin();
	for(int i=0; i<index; i++) { it++; }
	auto n_node = new Node<type>(std::forward<type>(obj), parent._node);
	auto pos = parent._node->_children.insert(it, n_node);
	n_node->_iter = pos;
	_size++;
	return Iterator<type>(this, n_node); }

template<typename type>
Iterator<type> Drzewo<type>::erase(Iterator<type> element)
throw(std::invalid_argument, std::out_of_range) {
	if(element.get_tree() != this) {
		throw std::invalid_argument("given iterator points to another tree"); }
	if(element == end()) {
		throw std::out_of_range("iterator out of tree"); }
	if(element.getNumberOfChildren()>0) {
		throw std::invalid_argument("ereased node cannot have any child"); }
	Node<type>* dying = element._node;
	if(element._node->_parent) {
		auto it_in_parent_list = dying->get_iter();
		dying->_parent->_children.erase(it_in_parent_list); }
	else {
		_root = nullptr; }
	element._node = dying->_parent;
	delete(dying);
	_size--;
	return element; }

template<typename type>
void Drzewo<type>::clear(Node<type>* node) {
	if(!node) {
		node = _root; }
	if(!node) {
		return; }
	while(!node->_children.empty()) {
		clear(node->_children.front()); }
	erase(Iterator<type>(this, node)); }


/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
// ITERATOR

template<typename type>
Iterator<type> Iterator<type>::getChild(int index) throw(std::out_of_range) {
	if(!_node) {
		throw std::out_of_range("iterator out of tree"); }
	if(index > _node->_children.size()) {
		throw std::out_of_range("given node doesn't have as many children"); }
	auto iter = _node->_children.begin();
	for(int i=0; i<index; i++) {
		++iter; }
	return Iterator<type>(_tree, *iter); }

template<typename type>
Iterator<type>& Iterator<type>::operator++() throw(std::out_of_range) {
	if(!_node) {
		throw std::out_of_range("iterator out of tree"); }
	if(!_node->_children.empty()) {
		_node = _node->_children.front();
		return *this; }
	while(_node->_parent && _node == _node->_parent->_children.back()) {
		_node = _node->_parent; }
	if(_node->_parent) {
		_node = *++(_node->get_iter()); }
	else {
		_node = nullptr; }
	return *this; }

template<typename type>
Iterator<type> Iterator<type>::operator++(int) throw(std::out_of_range) {
	auto iter = *this;
	++*this;
	return iter; }

template<typename type>
Iterator<type>& Iterator<type>::operator--() throw(std::out_of_range) {
	if(!_node) {
		throw std::out_of_range("iterator out of tree"); }
	if(_node == _tree->_root) {
		_node = nullptr;
		return *this; }
	else if(_node != _node->_parent->_children.front()) {
		_node = *--(_node->get_iter());
		while(!_node->_children.empty()) {
			_node = _node->_children.back(); } }
	else {
		_node = _node->_parent; }
	return *this; }

template<typename type>
Iterator<type> Iterator<type>::operator--(int) throw(std::out_of_range) {
	auto iter = *this;
	--*this;
	return iter; }
