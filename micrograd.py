import math

class Value:

  # General methods relating Value object
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0

    self._prev = set(_children)
    self._backward = lambda : None

    self._op = _op
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data})"


  # Operations that are used in application
  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward

    return out

  def __radd__(self, other):    # When python can't find __add__ in the left side object, it searches for __radd__ in the right side object.
    return self + other

  def __mul__(self, other):
      other = other if isinstance(other, Value) else Value(other)
      out = Value(self.data * other.data, (self, other), '*')

      def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
      out._backward = _backward

      return out

  def __rmul__(self, other):
    return self * other

  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)

  def __pow__(self, other):
    assert isinstance(other, (int, float))    # Other can only be a int or a float in this implementation. This may be changed for your needs
    out = Value(self.data ** other, (self, ), f"**{other}")

    def _backward():
      self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward

    return out

  def __truediv__(self, other):
    return self * other**-1
  
  def __exp__(self):
    out = Value(math.exp(self.data), (self, ), 'exp')

    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward

    return out

  # ------------------
  # Special Operations: If you would like to support any non-linear functions such as tanh or sigmoid, you can implement them HERE.
  # ------------------

  # Automize backward propogation.
  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    
    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()
