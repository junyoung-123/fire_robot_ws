"""Local static-scan registration for robot motion, not door-angle truth.

The reference consists solely of measured LiDAR returns. A visually observed
moving panel is excluded from this reference; odometry is only an ICP seed.
"""
from dataclasses import dataclass
from collections import deque
import math
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class ScanPose:
    pose: np.ndarray
    residual_rms_m: float
    translation_uncertainty_m: float
    yaw_uncertainty_rad: float
    inliers: int


def transform_cloud(points, pose):
    points,pose=np.asarray(points,float),np.asarray(pose,float)
    c,s=math.cos(pose[2]),math.sin(pose[2])
    return points@np.array([[c,-s],[s,c]]).T+pose[:2]


def compose_pose(left, right):
    left, right = np.asarray(left, float), np.asarray(right, float)
    angle = left[2]+right[2]
    return np.r_[transform_cloud(right[None, :2], left)[0],
                 math.atan2(math.sin(angle), math.cos(angle))]


def inverse_pose(pose):
    pose = np.asarray(pose, float)
    angle = -pose[2]
    c, s = math.cos(angle), math.sin(angle)
    return np.r_[-np.array([[c, -s], [s, c]])@pose[:2], angle]


def temporal_static_reference(samples, minimum=17, window=2.):
    """Pool stationary, original beam observations before arm contact.

    This is a localization reference, not a free-space or collision filter.
    Unknown, changing and insufficiently supported beams remain excluded.
    """
    if not samples:
        raise ValueError('Awaiting stationary measured scan reference')
    end=samples[-1][0]
    rows=[sample for sample in samples if 0.<=end-sample[0]<=window]
    stamps=np.array([sample[0] for sample in rows],float)
    if (len(rows)<minimum or not np.all(np.isfinite(stamps))
            or np.any(np.diff(stamps)<=0.) or stamps[-1]-stamps[0]<1.4):
        raise ValueError('Awaiting stationary measured scan reference')
    poses=np.asarray([sample[1] for sample in rows],float)
    if poses.shape!=(len(rows),3) or not np.all(np.isfinite(poses)):
        raise ValueError('Invalid measured reference poses')
    delta=poses-poses[-1]
    delta[:,2]=np.arctan2(np.sin(delta[:,2]),np.cos(delta[:,2]))
    if np.max(np.linalg.norm(delta[:,:2],axis=1))>.005 or np.max(abs(delta[:,2]))>.003:
        raise ValueError('Initial scan reference requires a stationary robot')
    clouds=np.asarray([sample[2] for sample in rows],float)
    if clouds.ndim!=3 or clouds.shape[0]!=len(rows) or clouds.shape[2]!=2:
        raise ValueError('Invalid aligned observed reference beams')
    finite=np.all(np.isfinite(clouds),axis=2)
    support=finite.sum(axis=0)>=math.ceil(.8*len(rows))
    points=np.nanmedian(np.where(finite[:,:,None],clouds,np.nan)[:,support],axis=0)
    deviations=np.linalg.norm(clouds[:,support]-points,axis=2)
    stable=np.nanmedian(deviations,axis=0)<=.06
    points=points[stable]
    if len(points)<60 or not np.all(np.isfinite(points)):
        raise ValueError('Too few temporally supported static reference beams')
    return points


def exclude_panel(points, segment, padding=.22):
    points, segment = np.asarray(points, float), np.asarray(segment, float)
    if segment.shape != (2, 2) or not np.all(np.isfinite(segment)):
        raise ValueError('Invalid independently observed panel segment')
    axis = segment[1]-segment[0]
    length = np.linalg.norm(axis)
    if length < .1:
        raise ValueError('Observed panel segment is too short')
    axis /= length
    local = points-segment[0]
    along = local@axis
    across = abs(local@np.array([-axis[1], axis[0]]))
    selected = (along > -padding)&(along < length+padding)&(across < padding)
    return points[~selected]


def exclude_observed_panel(points,panel,padding=.22):
    """Occlusion may shorten the visible patch, not invalidate its fit plane.

    A qualified short patch removes the whole measured plane band. This
    discards extra returns rather than inventing static localization support.
    """
    try:
        return exclude_panel(points,panel['segment_xy'],padding)
    except ValueError as exc:
        if str(exc)!='Observed panel segment is too short':
            raise
    points=np.asarray(points,float)
    center=np.asarray(panel.get('center'),float)
    normal=np.asarray(panel.get('normal'),float)
    uncertainty=panel.get('orientation_uncertainty_rad')
    residual=panel.get('residual_m')
    numeric=all(isinstance(value,(int,float,np.number)) for value in (uncertainty,residual))
    if (not panel.get('valid') or center.shape!=(3,) or normal.shape!=(3,)
            or not np.all(np.isfinite(center)) or not np.all(np.isfinite(normal))
            or not numeric or not math.isfinite(uncertainty) or not 0.<=uncertainty<=math.radians(2.)
            or not math.isfinite(residual) or not 0.<=residual<=.03
            or abs(np.linalg.norm(normal)-1.)>1.e-6 or abs(normal[2])>.15):
        raise ValueError('Short observed panel lacks qualified plane support')
    axis=normal[:2]/np.linalg.norm(normal[:2])
    return points[abs((points-center[:2])@axis)>=padding]


def exclude_observed_plane(points,panel,padding=.22):
    """Conservative moving-plane exclusion for new static map observations."""
    center=np.asarray(panel.get('center'),float)
    normal=np.asarray(panel.get('normal'),float)
    uncertainty=panel.get('orientation_uncertainty_rad')
    residual=panel.get('residual_m')
    if (not panel.get('valid') or center.shape!=(3,) or normal.shape!=(3,)
            or not np.all(np.isfinite(center)) or not np.all(np.isfinite(normal))
            or not isinstance(uncertainty,(int,float,np.number))
            or not isinstance(residual,(int,float,np.number))
            or not math.isfinite(uncertainty) or not 0.<=uncertainty<=math.radians(2.)
            or not math.isfinite(residual) or not 0.<=residual<=.03
            or abs(np.linalg.norm(normal)-1.)>1.e-6 or abs(normal[2])>.15):
        raise ValueError('Static extension requires a qualified moving panel plane')
    axis=normal[:2]/np.linalg.norm(normal[:2])
    points=np.asarray(points,float)
    return points[abs((points-center[:2])@axis)>=padding]


class ObservedStaticExtension:
    """Append measured structure without replacing or recursively drifting anchors.

    Every contributing pose must be independently fitted to the original frozen
    reference, never to this extension. Repeated timestamps are not evidence.
    """
    def __init__(self,reference):
        self.points=np.asarray(reference,float).copy()
        self.rows=deque(maxlen=24)
        self.last_stamp=-math.inf
        self.last_commit=-math.inf
        self.yaw_uncertainty_rad=0.

    def observe(self,stamp,points,*,translation_uncertainty_m,yaw_uncertainty_rad):
        points=np.asarray(points,float)
        if (not math.isfinite(stamp) or stamp<=self.last_stamp
                or points.ndim!=2 or points.shape[1]!=2 or not np.all(np.isfinite(points))
                or not math.isfinite(translation_uncertainty_m)
                or not 0.<=translation_uncertainty_m<=.025
                or not math.isfinite(yaw_uncertainty_rad)
                or not 0.<=yaw_uncertainty_rad<=math.radians(.5)):
            return None
        self.last_stamp=stamp
        self.rows.append((stamp,points.copy(),yaw_uncertainty_rad))
        rows=[r for r in self.rows if 0.<=stamp-r[0]<=1.5]
        if (len(rows)<8 or rows[-1][0]-rows[0][0]<.6
                or stamp-self.last_commit<1.):
            return None
        combined=np.vstack([r[1] for r in rows])
        if len(combined)<60:
            return None
        cells=np.floor(combined/.08).astype(int)
        _,groups=np.unique(cells,axis=0,return_inverse=True)
        candidates=np.array([np.median(combined[groups==i],axis=0)
                             for i in range(groups.max()+1)])
        support=np.zeros(len(candidates),int)
        for _,cloud,_ in rows:
            if len(cloud):
                support+=cKDTree(cloud).query(candidates)[0]<=.08
        distance=cKDTree(self.points).query(candidates)[0]
        new=candidates[(support>=math.ceil(.8*len(rows)))&(distance>.15)]
        if len(new)<20:
            return None
        # Rebuilding may change line normals, but old measured XY anchors stay
        # byte-identical. Validate the candidate before committing anything.
        points=np.vstack((self.points,new))
        registration=StaticScanRegistration(points)
        self.points=points
        self.last_commit=stamp
        self.yaw_uncertainty_rad=max(self.yaw_uncertainty_rad,max(r[2] for r in rows))
        return registration,dict(new_measured_points=len(new),
            original_measurement_stamps=[r[0] for r in rows],
            total_measured_points=len(points),
            extension_anchor_yaw_uncertainty_rad=self.yaw_uncertainty_rad,
            basis='repeated_static_returns_registered_only_to_original_reference',
            moving_plane_excluded=True,old_reference_coordinates_unchanged=True)


class CorrectionFilter:
    """Slow slip correction; wheel motion is not low-pass filtered."""
    def __init__(self, initial, time_constant=2., yaw_time_constant=None):
        self.pose = np.asarray(initial, float).copy()
        self.stamp = None
        self.time_constant = time_constant
        self.yaw_time_constant=time_constant if yaw_time_constant is None else yaw_time_constant

    def update(self, correction, stamp):
        correction = np.asarray(correction, float)
        if (correction.shape != (3,) or not np.all(np.isfinite(correction))
                or not math.isfinite(stamp) or self.time_constant <= 0.
                or not math.isfinite(self.yaw_time_constant) or self.yaw_time_constant<=0.
                or (self.stamp is not None and stamp <= self.stamp)):
            raise ValueError('Invalid scan correction filter observation')
        if self.stamp is None:
            self.pose = correction.copy()
        else:
            weight = -math.expm1(-(stamp-self.stamp)/self.time_constant)
            delta = correction-self.pose
            delta[2] = math.atan2(math.sin(delta[2]), math.cos(delta[2]))
            yaw_weight=-math.expm1(-(stamp-self.stamp)/self.yaw_time_constant)
            self.pose[:2] += weight*delta[:2]
            self.pose[2] += yaw_weight*delta[2]
            self.pose[2] = math.atan2(math.sin(self.pose[2]), math.cos(self.pose[2]))
        self.stamp = stamp
        return self.pose.copy()

    def update_at_sensor_pivot(self,measured_pose,wheel_pose,stamp,*,apply_translation):
        """Separate yaw correction from translation at the current sensor.

        Filtering SE(2) correction components about the odometry origin turns
        a small yaw update into a large fictitious translation far from spawn.
        Preserve wheel motion and filter only measured local translation error.
        """
        measured_pose=np.asarray(measured_pose,float)
        wheel_pose=np.asarray(wheel_pose,float)
        if (measured_pose.shape!=(3,) or wheel_pose.shape!=(3,)
                or not np.all(np.isfinite(measured_pose)) or not np.all(np.isfinite(wheel_pose))):
            raise ValueError('Invalid timestamped sensor pivot poses')
        previous_pose=compose_pose(self.pose,wheel_pose)
        previous_stamp=self.stamp
        measured_correction=compose_pose(measured_pose,inverse_pose(wheel_pose))
        corrected=self.update(measured_correction,stamp)
        weight=1. if previous_stamp is None else -math.expm1(-(stamp-previous_stamp)/self.time_constant)
        position=previous_pose[:2].copy()
        if apply_translation:
            position+=weight*(measured_pose[:2]-position)
        heading=compose_pose(corrected,wheel_pose)[2]
        self.pose=compose_pose(np.r_[position,heading],inverse_pose(wheel_pose))
        return self.pose.copy()


class StaticScanRegistration:
    def __init__(self, reference):
        points=np.asarray(reference,float)
        if points.ndim!=2 or points.shape[1]!=2 or len(points)<60 or not np.all(np.isfinite(points)):
            raise ValueError('Insufficient finite observed static scan reference')
        # Local line normals preserve geometric scan structure instead of
        # letting dense angular sampling dominate point-to-point matching.
        tree=cKDTree(points)
        distance,index=tree.query(points,k=9)
        neighbors=points[index]
        centered=neighbors-neighbors.mean(axis=1,keepdims=True)
        covariance=np.einsum('nki,nkj->nij',centered,centered)/9.
        values,vectors=np.linalg.eigh(covariance)
        lines=(values[:,0]<.25*np.maximum(values[:,1],1.e-8))&(distance[:,-1]<.35)
        if lines.sum()<60:
            raise ValueError('Observed static reference lacks line support')
        self.points=points[lines]
        self.normals=vectors[lines,:,0]
        self.tree=cKDTree(self.points)

    def estimate(self, cloud, initial, *, require_full_translation=True):
        # The opt-in angular consumer must not apply uncertified translation.
        # Residual, geometric observability and yaw limits remain mandatory.
        cloud,pose=np.asarray(cloud,float),np.asarray(initial,float).copy()
        if (cloud.ndim!=2 or cloud.shape[1]!=2 or len(cloud)<60 or pose.shape!=(3,)
                or not np.all(np.isfinite(cloud)) or not np.all(np.isfinite(pose))):
            raise ValueError('Invalid measured local scan inputs')
        final=None
        for _ in range(20):
            moved=transform_cloud(cloud,pose)
            distances,index=self.tree.query(moved)
            normals=self.normals[index]
            residual=np.einsum('ij,ij->i',moved-self.points[index],normals)
            selected=(distances<.22)&(abs(residual)<.12)
            if selected.sum()<60:
                raise ValueError('Too few observed static scan correspondences')
            residual=residual[selected]
            xy=moved[selected]-pose[:2]
            n=normals[selected]
            jacobian=np.column_stack((n[:,0],n[:,1],-xy[:,1]*n[:,0]+xy[:,0]*n[:,1]))
            # Cauchy residual weights retain thin fixed structures while
            # rejecting independently moving or mismatched returns.
            weight=1./(1.+(residual/.035)**2)
            system=jacobian.T@(weight[:,None]*jacobian)
            eigen=np.linalg.eigvalsh(system)
            if eigen[0]<1.e-3 or eigen[-1]/eigen[0]>1.e5:
                raise ValueError('Static scan motion is geometrically unobservable')
            step=-np.linalg.solve(system,jacobian.T@(weight*residual))
            if np.linalg.norm(step[:2])>.10 or abs(step[2])>.08:
                raise ValueError('Static registration step exceeds local robot motion budget')
            pose+=step
            pose[2]=math.atan2(math.sin(pose[2]),math.cos(pose[2]))
            rms=math.sqrt(float(np.sum(weight*residual**2)/np.sum(weight)))
            covariance=np.linalg.inv(system)*max(.005,rms)**2
            uncertainty=3.*math.sqrt(float(np.linalg.eigvalsh(covariance[:2,:2])[-1]))
            yaw_uncertainty=3.*math.sqrt(float(covariance[2,2]))
            final=ScanPose(pose.copy(),rms,uncertainty,yaw_uncertainty,int(selected.sum()))
            if np.linalg.norm(step[:2])<.0001 and abs(step[2])<.0001:
                break
        if (final.residual_rms_m>.055 or final.yaw_uncertainty_rad>math.radians(2.)
                or (require_full_translation and final.translation_uncertainty_m>.025)):
            raise ValueError('Static scan pose has unbounded measurement uncertainty')
        return final
