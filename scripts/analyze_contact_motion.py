"""Compare commanded motion with recorded odometry and passive contact pairs."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(folder):
    data = json.loads((folder/'motion_diagnostics.json').read_text())
    result = json.loads((folder/'result.json').read_text())
    contacts = json.loads((folder/'contact_diagnostics.json').read_text())
    odom = [x for x in data['odom'] if x['phase'] == 'HANDLE_HELD_BASE_OPEN']
    if not odom:
        raise ValueError('No held-opening odometry')
    end = odom[-1]['wall_elapsed_sec']
    tail = [x for x in odom if x['wall_elapsed_sec'] >= end-15.]
    start = tail[0]['wall_elapsed_sec']
    commands = [x for x in data['commands'] if start <= x['wall_elapsed_sec'] <= end]
    pairs = sorted({(x['part'], c['collision1'], c['collision2'])
                    for x in contacts['samples'] if start <= x['wall_elapsed_sec'] <= end
                    for c in x['contacts']})
    joints = [x for x in data['joints'] if start <= x['wall_elapsed_sec'] <= end]
    joint_motion = {}
    if joints:
        for name in joints[0]['names']:
            values = [x['positions'][x['names'].index(name)] for x in joints
                      if name in x['names'] and len(x['positions']) == len(x['names'])]
            joint_motion[name] = max(values)-min(values) if values else None
    def angles(item):
        x,y,z,w=item['quaternion']
        return [math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                math.asin(max(-1., min(1., 2*(w*y-z*x)))),
                math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))]
    rpy=np.array([angles(x) for x in tail])
    summary = dict(scope='Last 15 wall seconds of recorded held-opening phase; passive diagnostics.',
        run=folder.name, wall_window=[start,end],
        base_translation_m=float(np.linalg.norm(np.array(tail[-1]['xyz'])-tail[0]['xyz'])),
        base_rpy_range_deg=np.degrees(np.ptp(rpy,axis=0)).tolist(),
        base_last_rpy_deg=np.degrees(rpy[-1]).tolist(),
        base_z_range_m=[min(x['xyz'][2] for x in tail), max(x['xyz'][2] for x in tail)],
        commanded_positive_samples={topic:sum(x['topic']==topic and x['linear_x']>.005 for x in commands)
                                    for topic in ('/cmd_vel_manual','/cmd_vel','/cmd_vel_safe')},
        contact_pairs=pairs, joint_position_range=joint_motion)
    (folder/'motion_review.json').write_text(json.dumps(summary,indent=2))
    fig, axes=plt.subplots(4,1,figsize=(11,10),sharex=True,layout='constrained')
    for topic in ('/cmd_vel_manual','/cmd_vel','/cmd_vel_safe'):
        cc=[x for x in data['commands'] if x['topic']==topic]
        axes[0].plot([x['wall_elapsed_sec'] for x in cc],[x['linear_x'] for x in cc],label=topic,alpha=.7)
    axes[0].set_ylabel('Command m/s');axes[0].legend()
    t=np.array([x['wall_elapsed_sec'] for x in odom])
    xyz=np.array([x['xyz'] for x in odom])
    displacement=np.linalg.norm(xyz-xyz[0],axis=1)
    axes[1].plot(t,displacement,label='Measured base translation')
    axes[1].set_ylabel('Displacement m');axes[1].legend()
    events=[x for x in result['feedback_events'] if x.get('event')=='handle_held_base_sample']
    axes[2].plot([x['wall_elapsed_sec'] for x in events],
                 [math.degrees(x['door_delta_rad']) for x in events],label='Door opening')
    axes[2].axhline(math.degrees(result['min_angle_rad']),color='#b22b31',linestyle='--',label='Required angle')
    axes[2].set_ylabel('Door degrees');axes[2].legend()
    all_rpy=np.degrees(np.array([angles(x) for x in odom]))
    for i,name in enumerate(('roll','pitch','yaw')):
        axes[3].plot(t,all_rpy[:,i],label=name)
    axes[3].set_ylabel('Base degrees');axes[3].set_xlabel('Wall elapsed seconds');axes[3].legend()
    for ax in axes:
        ax.axvspan(start,end,color='#e8dba6',alpha=.3)
        ax.grid(alpha=.2)
    fig.suptitle('Command, motion and contact diagnosis | '+folder.name,fontsize=12)
    fig.savefig(folder/'motion_review.png',dpi=140)
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    main(parser.parse_args().folder)
