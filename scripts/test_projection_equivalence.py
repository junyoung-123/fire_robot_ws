from itertools import combinations
import unittest
import numpy as np
from fire_robot_manipulation.contact_feedback import _nearest_velocity


def original_projection(requested, rows, bounds, metric):
    inverse = np.linalg.inv(metric)
    candidates = [requested]
    for row,bound in zip(rows,bounds):
        direction = inverse@row
        divisor = direction@row
        if divisor > 1.e-12:
            candidates.append(requested+(bound-row@requested)*direction/divisor)
    for first,second in combinations(range(len(rows)),2):
        matrix = rows[[first,second]]
        if abs(np.linalg.det(matrix)) > 1.e-10:
            candidates.append(np.linalg.solve(matrix,bounds[[first,second]]))
    candidates = [p for p in candidates if np.all(rows@p <= bounds+1.e-9)]
    if not candidates:
        return None
    cost = lambda p: float((p-requested)@metric@(p-requested))
    best = min(candidates,key=cost)
    return cost(best),best


class ProjectionEquivalenceTests(unittest.TestCase):
    def test_vectorized_candidates_match_original_exact_solver(self):
        rng = np.random.default_rng(407)
        box = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        for _ in range(70):
            rows = np.vstack((box,rng.normal(size=(18,2))))
            bounds = np.r_[[.06,.04,.08,.08],rng.uniform(0.,.1,18)]
            factor = rng.normal(size=(2,2))
            metric = factor.T@factor+np.diag([.1,.2])
            requested = rng.normal(size=2)*.15
            old = original_projection(requested,rows,bounds,metric)
            new = _nearest_velocity(requested,rows,bounds,.6,metric)
            self.assertIsNotNone(new)
            self.assertAlmostEqual(old[0],new[0],places=11)
            np.testing.assert_allclose(old[1],new[1],atol=1.e-9)

    def test_infeasible_parallel_halfplanes_still_rejected(self):
        rows = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        self.assertIsNone(_nearest_velocity(np.zeros(2),rows,np.array([-.1,0.,.08,.08]),.6))

    def test_feasible_request_is_unchanged(self):
        point = np.array([.01,.02])
        result = _nearest_velocity(point,np.eye(2),np.ones(2),.6)
        np.testing.assert_array_equal(result[1],point)
        self.assertEqual(result[0],0.)


if __name__ == '__main__':
    unittest.main()
