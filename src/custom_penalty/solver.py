import warnings
import numpy as np

from typing import Callable, Any, Literal
from collections.abc import Sequence
from scipy.optimize import minimize, OptimizeResult

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.primitives import BaseSamplerV2
from qiskit.primitives import BackendSamplerV2 as BackendSampler
from qiskit.circuit import ParameterVector

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import GurobiOptimizer
from qiskit_algorithms.optimizers import OptimizerResult
from qiskit_aer import AerSimulator


TWO_PI = 2 * np.pi

class CustomPenaltySolver:
    def __init__(
        self,
        quadratic_program: QuadraticProgram,
        ansatz: QuantumCircuit | None = None,
        penalty_mult: Sequence[float] | None = None,
        penalty_func: Callable | None = None,
        default_shots: int = 1000,
        filter_count_threshold: float = 0.0,
        sampler: BaseSamplerV2 | None = None,
        save_params_history: bool = True,
    ):
        self.qp = quadratic_program
        self.num_qubits = self.qp.get_num_binary_vars()
        self.num_constraints = self.qp.get_num_linear_constraints() + self.qp.get_num_quadratic_constraints()

        # set default penalty using upper bound value
        default_penalty_mult = self.qp.objective.evaluate(np.ones(self.num_qubits)) * 2
        self.penalty_mult = [default_penalty_mult] * self.num_constraints if penalty_mult is None else penalty_mult

        # assume <= 0 satisfies the constraint
        self.penalty_func = (lambda x: np.heaviside(x, 0)) if penalty_func is None else penalty_func

        if ansatz is None:
            self.set_ansatz(self.default_ansatz(self.num_qubits))
        else:
            self.set_ansatz(ansatz)

        backend_mps = AerSimulator(method='matrix_product_state')
        self.sampler = BackendSampler(backend=backend_mps) if sampler is None else sampler
        self.default_shots = default_shots
        self.cvar_options = {
            'alpha': 0.1,
            'shots': default_shots,
        }
        self.filter_count_threshold = filter_count_threshold
        self._save_params_history = save_params_history

        self.result: OptimizeResult | None = None
        self.optimal_params = None
        self.optimal_solution: list[int] | None = None
        self.optimal_counts: dict[str, int] | None = None
        self.sorted_elc: list[tuple[str, float, int]] | None = None
        self.optimal_prob: float | None = None
        self.cvar_optimal_prob: float | None = None
        self.highest_prob_elc: tuple[str, float, int] | None = None
        self.obj_history: list[float] = []
        self.params_history: list[np.ndarray] = []

        self._min_exp = np.inf

    @property
    def ansatz(self) -> QuantumCircuit:
        return self._ansatz

    def set_ansatz(self, new_ansatz: QuantumCircuit):
        if new_ansatz.num_qubits != self.num_qubits:
            raise ValueError(f'The number of qubits in the ansatz must be {self.num_qubits}, got {new_ansatz.num_qubits} instead.')
        self._ansatz = new_ansatz

    @property
    def optimal_params(self) -> np.ndarray | None:
        if not self._save_params_history:
            warnings.warn('Parameter history saving is turned off. Optimal parameters might not tally with the min. objective saved.')
        return self._optimal_params

    @optimal_params.setter
    def optimal_params(self, new_value: np.ndarray | None):
        self._optimal_params = new_value

    @property
    def penalty_mult(self) -> Sequence[float]:
        return self._penalty_mult

    @penalty_mult.setter
    def penalty_mult(self, new_penalty_mult: Sequence[float]):
        assert len(new_penalty_mult) == self.num_constraints, \
            f'Penalty factors must be a scalar (global penalty) or a sequence of length {self.num_constraints} (no. of constraints).'
        self._penalty_mult = new_penalty_mult

    @staticmethod
    def default_ansatz(
        num_qubits: int,
        reps: int = 1,
        entanglement: Literal['linear', 'circular'] = 'linear',
    ):
        # only linear or circular entanglement,
        # as we encourage fast simulation with MPS, so the circuit cannot be too dense.
        if entanglement not in ['linear', 'circular']:
            raise ValueError(f'Entanglment must be linear or circular, got {entanglement} instead.')

        qc = QuantumCircuit(num_qubits)
        params = ParameterVector('θ', num_qubits * (reps + 1))

        for i in range(num_qubits):
            qc.ry(params[i], i)

        for r in range(1, reps + 1):
            for i in range(num_qubits - 1):
                qc.cz(i, i+1)
            if entanglement == 'circular':
                qc.cz(num_qubits - 1, 0)
            for i in range(num_qubits):
                qc.ry(params [r * num_qubits + i], i)

        return qc

    def generate_random_params(self, scale: float = TWO_PI) -> np.ndarray:
        return np.random.rand(self.ansatz.num_parameters) * scale

    def compute_classical_loss(self, solution: list[int]) -> float:
        obj_val = self.qp.objective.evaluate(solution)
        if self.qp.objective.sense == self.qp.objective.sense.MAXIMIZE:
            # negate if maximization problem
            obj_val = -obj_val

        constraints = self.qp.linear_constraints + self.qp.quadratic_constraints
        constr_vals = []
        for i, constraint in enumerate(constraints):
            if constraint.sense == constraint.sense.EQ:
                constr_vals.append(self.penalty_mult[i] * (constraint.rhs - constraint.evaluate(solution))**2)
            elif constraint.sense == constraint.sense.LE:
                constr_vals.append(self.penalty_mult[i] * self.penalty_func(constraint.evaluate(solution) - constraint.rhs))
            elif constraint.sense == constraint.sense.GE:
                constr_vals.append(self.penalty_mult[i] * self.penalty_func(constraint.rhs - constraint.evaluate(solution)))
            else:
                raise ValueError(f'Unknown sense {constraint.sense}.')

        return obj_val + sum(constr_vals)

    def filter_counts(self, counts: dict[str, int]) -> dict[str, int]:
        # threshold is exclusive, i.e. threshold will not be included in the filtered counts.

        return {k: v for k, v in counts.items() if (v / sum(counts.values())) > self.filter_count_threshold}

    def compute_finite_sampling_loss(self, params) -> float:
        ansatz = self.ansatz.copy()
        ansatz.measure_all()

        result = self.sampler.run([(ansatz, params)], shots=self.default_shots).result()[0]

        counts = result.data.meas.get_counts()
        counts = self.filter_counts(counts)

        expectation = 0
        for eigenstate, count in counts.items():
            solution = list(map(int, eigenstate))[::-1] # reverse due to qiskit ordering
            loss = self.compute_classical_loss(solution)
            expectation += loss * count

        expectation /= self.default_shots

        if expectation < self._min_exp:
            self._min_exp = expectation
            self.optimal_solution = self.sample_most_likely(counts)

        return expectation

    def compute_cvar_loss(self, params) -> float:
        ansatz = self.ansatz.copy()
        ansatz.measure_all()

        alpha = self.cvar_options['alpha']
        shots = self.cvar_options['shots']

        result = self.sampler.run([(ansatz, params)], shots=shots).result()[0]

        counts = result.data.meas.get_counts()
        counts = self.filter_counts(counts)
        num_shots_after_filter = sum(counts.values())
        num_samples = np.ceil(alpha * num_shots_after_filter)

        es_loss_count = []
        for eigenstate, count in counts.items():
            solution = list(map(int, eigenstate))[::-1] # reverse due to qiskit ordering
            loss = self.compute_classical_loss(solution)
            es_loss_count.append((eigenstate, loss, count))

        sorted_elc = sorted(es_loss_count, key=lambda x: x[1]) # sort based on the lowest loss first

        # store data
        self.sorted_elc = sorted_elc
        self.optimal_counts = counts

        # compute cvar
        sctr = 0
        cvar_loss = 0
        for es, loss, count in sorted_elc:
            if sctr + count > num_samples:
                rem = num_samples - sctr
                cvar_loss += loss * rem
                break
            else:
                cvar_loss += loss * count
                sctr += count

        cvar_loss /= num_samples
        return cvar_loss

    def get_state(self, params) -> Statevector:
        param_qc = self.ansatz.assign_parameters(params)
        sv = Statevector(param_qc)
        return sv

    def _sample_optimal_circuit(self) -> dict[str, int]:
        """
        Run the full circuit with sampler to get the solution.
        """
        if self.optimal_params is None:
            raise ValueError(f'Problem not yet solved. Run {type(self).__name__}.solve() to solve the problem.')

        ansatz = self.ansatz.copy()
        ansatz.measure_all()

        result = self.sampler.run([(ansatz, self.optimal_params)], shots=self.default_shots).result()[0]
        return result.data.meas.get_counts()

    def sample_most_likely(self, counts: dict[str, int]) -> list[int]:
        highest_count = max(counts.values())
        self.optimal_prob = highest_count / self.default_shots

        most_likely_bs = max(counts, key=counts.get)
        return list(map(int, most_likely_bs[::-1])) # flip the bit-string due to qiskit ordering

    def get_optimal_solution(self) -> list[int]:
        counts = self._sample_optimal_circuit()
        self.optimal_counts = counts
        return self.sample_most_likely(counts)

    def solve(
        self,
        initial_point: np.ndarray | None = None,
        sampling_method: Literal['fs', 'cvar'] = 'fs',
        optimizer: str = 'powell',
        optimizer_options: dict[str, Any] | None = None,
        cvar_options: dict[str, Any] | None = None,
        shots: int | None = None,
        filter_count_threshold: float | None = None,
    ) -> OptimizerResult:
        """
        Calls the Scipy minimize function and returns the OptimizerResult object.
        Methods available:
          - 'fs': Finite sampling; evaluates losses classically and multiply by the probability counts obtained.
          - 'cvar': Computes the CVaR (Conditional Value at Risk) of the given loss function, similar to finite sampling but with loss that only considers
                    a certain percentage of best shots. The parameters can be set by passing `cvar_options`.
        """
        if shots is not None:
            self.default_shots = shots
        if filter_count_threshold is not None:
            self.filter_count_threshold = filter_count_threshold

        if initial_point is None:
            initial_point = self.generate_random_params()
        else:
            assert np.asarray(initial_point).shape[0] == self.ansatz.num_parameters, 'Parameter length does not match.'

        if cvar_options is not None:
            self.cvar_options.update(cvar_options)

        if sampling_method == 'fs':
            obj_func = self.compute_finite_sampling_loss
        elif sampling_method == 'cvar':
            obj_func = self.compute_cvar_loss
        else:
            raise ValueError('Unrecognized method. Method must be one of `matmul`, `fs` or `cvar`.')

        def callback(intermediate_result: OptimizeResult) -> None:
            self.obj_history.append(intermediate_result.fun)
            if self._save_params_history:
                self.params_history.append(intermediate_result.x)

        result = minimize(
            obj_func,
            initial_point,
            method=optimizer,
            options=optimizer_options,
            callback=callback,
        )

        self.result = result

        min_idx = np.argmin(self.obj_history)
        if self._save_params_history:
            self.optimal_params = self.params_history[min_idx]
        else:
            self.optimal_params = result.x

        if sampling_method == 'cvar':
            self.optimal_solution = list(map(int, self.sorted_elc[0][0]))[::-1]

            optimal_count = 0
            optimal_obj = self.sorted_elc[0][1]
            for es, loss, count in self.sorted_elc:
                if loss != optimal_obj:
                    break
                optimal_count += count
            self.optimal_prob = optimal_count / self.default_shots

            es, loss, count = max(self.sorted_elc, key=lambda x: x[2])
            self.highest_prob_elc = (es, loss, count/self.default_shots)
        else:
            # cvar will store optimal solution at every step of optimization
            self.optimal_solution = self.get_optimal_solution()

        return result

    def solve_gurobi(self) -> tuple[float, np.ndarray]:
        result = GurobiOptimizer().solve(self.qp)
        return result.fval, result.x