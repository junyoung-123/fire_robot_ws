import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fire_robot_perception.inference_runtime import bounded_yolo_inference


class InferenceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.count = 8
        self.torch = SimpleNamespace(get_num_threads=lambda: self.count,
                                     set_num_threads=lambda n: setattr(self, 'count', n))

    def test_lazy_backend_cannot_leave_cpu_pool_oversubscribed(self):
        def model(image, **kwargs):
            self.assertEqual(self.count, 2)
            self.count = 8
            return image, kwargs
        with patch.dict('sys.modules', torch=self.torch):
            result = bounded_yolo_inference(model, 'image', 2, conf=.2)
        self.assertEqual(result, ('image', {'conf': .2}))
        self.assertEqual(self.count, 2)

    def test_failed_inference_restores_budget_and_propagates_error(self):
        def model(*a):
            self.count = 8
            raise RuntimeError('prediction failed')
        with patch.dict('sys.modules', torch=self.torch):
            with self.assertRaises(RuntimeError):
                bounded_yolo_inference(model, None, 2)
        self.assertEqual(self.count, 2)

    def test_disabled_limit_preserves_existing_behavior(self):
        with patch.dict('sys.modules', torch=self.torch):
            self.assertEqual(bounded_yolo_inference(lambda image: image, 3, 0), 3)
        self.assertEqual(self.count, 8)


if __name__ == '__main__':
    unittest.main()
