import unittest
from check_manual_navigation_exclusion import check_text


class ExclusionTests(unittest.TestCase):
    def test_goal_during_escape_is_a_failure(self):
        log = '[INFO] [100.0] [state_machine_node]: Local obstacle escape 1/2: stuck\n'
        log += '[INFO] [102.0] [bt_navigator]: Begin navigating from current location (0,0)\n'
        result = check_text(log)
        self.assertEqual(result['new_goals_during_escape'][0]['line'], 2)

    def test_normal_goal_after_escape_is_allowed(self):
        log = '[INFO] [100.0] [state_machine_node]: Local obstacle escape 1/2: stuck\n'
        log += '[INFO] [102.0] [state_machine_node]: Local obstacle escape finished\n'
        log += '[INFO] [103.0] [bt_navigator]: Begin navigating from current location (0,0)\n'
        self.assertEqual(check_text(log)['new_goals_during_escape'], [])

    def test_no_escape_does_not_claim_escape_was_exercised(self):
        self.assertEqual(check_text('')['intervals'], 0)


if __name__ == '__main__':
    unittest.main()
