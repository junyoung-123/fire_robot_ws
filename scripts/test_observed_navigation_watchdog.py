import unittest

from run_observed_navigation import ReadinessWatch


class ReadinessWatchTests(unittest.TestCase):
    def ready(self):
        return dict(navigate_action_server_present=True, map_messages_recorded=2,
                    map_to_base_available=True)

    def test_missing_startup_still_fails(self):
        watch = ReadinessWatch()
        audit = self.ready()
        audit['map_to_base_available'] = False
        self.assertIsNone(watch.check(10, audit))
        self.assertEqual(watch.check(181, audit), 'SLAM_TRANSFORM_STARTUP_TIMEOUT')

    def test_transient_loss_after_startup_is_not_immediate_failure(self):
        watch = ReadinessWatch()
        audit = self.ready()
        self.assertIsNone(watch.check(30, audit))
        audit['map_to_base_available'] = False
        self.assertIsNone(watch.check(181, audit))
        self.assertIsNone(watch.check(220, audit))
        self.assertIsNone(watch.check(225, self.ready()))
        self.assertIsNone(watch.check(300, audit))

    def test_persistent_runtime_failure_still_stops_run(self):
        watch = ReadinessWatch()
        watch.check(30, self.ready())
        audit = self.ready()
        audit['navigate_action_server_present'] = False
        self.assertIsNone(watch.check(200, audit))
        self.assertEqual(watch.check(261, audit), 'NAV2_RUNTIME_TIMEOUT')


if __name__ == '__main__':
    unittest.main()
