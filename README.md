# Custom Penalty Solver
This is an implementation of solving constrained binary quadratic programs with custom penalties using the variational quantum eigensolver (VQE).
The original work is available at [arXiv:2604.20088](https://arxiv.org/abs/2604.20088).

# Getting started
We recommend using `uv` ([installation guide](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1)).
```
uv venv --python 3.12
uv pip install custom-penalty
```

Or you can go with the traditional:
```
python3.12 -m venv .venv
source .venv/bin/activate
pip install custom-penalty
```

# Usage
It is easier to define the problem with `docplex`, then convert it into a `QuadraticProgram`:
```
mdl = Model()

# define variables
x = mdl.binary_var_list(2, name='x')

# define objectives
mdl.maximize(x[0] + x[1] - x[0]*x[1])

# add constraints
mdl.add_constraint(x[0] + x[1] <= 1)
mdl.add_constraint(2*x[0] + 3*x[1] == 2)
mdl.add_constraint(5*x[0] + 6*x[1] >= 5)

# convert to quadratic program
qp = from_docplex_mp(mdl)
```

Then, you can solve it using
```
cps = CustomPenaltySolver(qp)
result = cps.solve()
print(result)
```

For customization and more detailed guide, refer [example.ipynb](example.ipynb).
