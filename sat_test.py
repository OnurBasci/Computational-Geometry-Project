from pysat.solvers import Solver

s = Solver(name='mc')

s.add_clause([1])
s.add_clause([1, 2])
s.add_clause([2, 3])

#s.add_atmost(lits=[1,2,3], k = 1, no_return=False) # returns false
s.add_atmost(lits=[1,2,3], k = 2, no_return=False) # returns true

print(s.solve())
print(s.get_model())