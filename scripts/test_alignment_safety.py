import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/fire_robot_fsm'))
from fire_robot_fsm.alignment_safety import grid_segment_is_clear
from fire_robot_fsm.fsm_subsystems import DoorApproachSubFsm


def grid():
    return NS(header=NS(frame_id='map'),data=[0]*100,
              info=NS(width=10,height=10,resolution=.1,
                      origin=NS(position=NS(x=0.,y=0.),orientation=NS(x=0.,y=0.,z=0.,w=1.))))


def method(name, globals_=None):
    tree=ast.parse((ROOT/'src/fire_robot_fsm/fire_robot_fsm/state_machine_node.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    space={'DoorInfo':object,'time':NS(monotonic=lambda:100.),
           'grid_segment_is_clear':grid_segment_is_clear,**(globals_ or {})}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'fsm_method','exec'),space)
    return space[name]


class AlignmentSafetyTests(unittest.TestCase):
    def test_free_and_lethal_endpoint(self):
        g=grid()
        self.assertTrue(grid_segment_is_clear(g,(.25,.25),(.75,.25)))
        for cost in (99,100,-1):
            g.data[27]=cost
            self.assertFalse(grid_segment_is_clear(g,(.25,.25),(.75,.25)))

    def test_blocked_segment_with_free_endpoints(self):
        g=grid();g.data[25]=100
        self.assertFalse(grid_segment_is_clear(g,(.25,.25),(.75,.25)))

    def test_invalid_geometry_and_outside(self):
        g=grid()
        self.assertFalse(grid_segment_is_clear(g,(-.001,.25),(.75,.25)))
        self.assertFalse(grid_segment_is_clear(g,(math.nan,.25),(.75,.25)))
        g.info.resolution=0
        self.assertFalse(grid_segment_is_clear(g,(.25,.25),(.75,.25)))

    def test_rotated_grid(self):
        g=grid();g.info.origin.orientation=NS(x=0.,y=0.,z=math.sqrt(.5),w=math.sqrt(.5))
        self.assertTrue(grid_segment_is_clear(g,(-.25,.25),(-.25,.75)))
        g.data[25]=100
        self.assertFalse(grid_segment_is_clear(g,(-.25,.25),(-.25,.75)))

    def test_rejected_goal_cannot_become_fine_alignment(self):
        node=NS(target_door=NS(door_color='blue'),_fine_alignment_path_clear=lambda d:False)
        for reason in ('nav2_failure','nav_success','door_nav_stuck'):
            self.assertFalse(DoorApproachSubFsm(node).can_continue_with_fine_alignment(reason))

    def test_missing_pose_is_not_ready(self):
        node=NS(_fine_alignment_path_clear=lambda d:True,_door_open_pose_error=lambda d:None)
        self.assertEqual(method('_prepare_door_opening_pose')(node,object()),'failed')

    def test_blocked_path_is_not_a_missing_visual_confirmation(self):
        stopped=[]
        node=NS(_fine_alignment_path_clear=lambda d:False,
                cmd_vel_pub=NS(publish=stopped.append),
                _door_open_align_start_time=object(),
                _door_open_fine_backoff_until=object())
        result=method('_prepare_door_opening_pose',{'Twist':NS})(node,object())
        self.assertEqual(result,'blocked')
        self.assertEqual(len(stopped),1)
        self.assertIsNone(node._door_open_align_start_time)

    def test_blocked_path_keeps_target_for_bounded_navigation_retry(self):
        target=NS(door_id='observed_blue_test',door_color='blue',confidence=.8)
        published=[]
        node=NS(target_door=target,_nav_retry_count=0,_max_nav_retries=3,
                get_logger=lambda:NS(warn=lambda *a,**k:None,info=lambda *a,**k:None),
                _door_has_map_identity=lambda d:True,
                _is_pre_exit_blue_failure_candidate=lambda xy:True,
                _door_identity_xy=lambda d:(0,0),
                _transition=lambda state:None,target_door_pub=NS(publish=published.append))
        import copy
        method('_handle_nav_failure',{'copy':copy,'State':NS(NAVIGATING=2)})(node,'alignment_path_blocked')
        self.assertIs(node.target_door,target)
        self.assertEqual(node._nav_retry_count,1)
        self.assertEqual(published[0].door_id,target.door_id)
        self.assertLess(published[0].confidence,0)

    def test_stale_or_wrong_frame_costmap_refuses_handoff(self):
        g=grid();door=NS(door_pose=NS(header=NS(frame_id='map'),pose=NS(position=NS(x=.75,y=.25))))
        node=NS(_alignment_costmap=(g,99.),_fine_alignment_costmap_max_age=3.,
                _current_map_pose=lambda:(.25,.25,0),
                get_clock=lambda:NS(now=lambda:NS(nanoseconds=int(100e9))))
        node._alignment_costmap_ready=lambda door:method('_alignment_costmap_ready')(node,door)
        check=method('_fine_alignment_path_clear')
        self.assertTrue(check(node,door))
        node._alignment_costmap=(g,90.)
        self.assertFalse(check(node,door))
        node._alignment_costmap=(g,99.);g.header.frame_id='odom'
        self.assertFalse(check(node,door))

    def test_costmap_publisher_matches_full_grid_consumer(self):
        config=yaml.safe_load((ROOT/'src/fire_robot_navigation/config/nav2_params.yaml').read_text())
        params=config['global_costmap']['global_costmap']['ros__parameters']
        self.assertTrue(params['always_send_full_costmap'])
        self.assertGreater(params['publish_frequency'],1/3)

    def test_clock_reset_refuses_old_snapshot(self):
        g=grid();door=NS(door_pose=NS(header=NS(frame_id='map')))
        node=NS(_alignment_costmap=(g,101.),_fine_alignment_costmap_max_age=3.,
                get_clock=lambda:NS(now=lambda:NS(nanoseconds=int(100e9))))
        self.assertFalse(method('_alignment_costmap_ready')(node,door))


if __name__=='__main__':
    unittest.main()
