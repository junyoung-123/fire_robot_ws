// Evaluation camera only. Never writes robot, door or lever components.
#include <chrono>
#include <cmath>
#include <string>
#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/Model.hh>
#include <ignition/gazebo/Util.hh>
#include <ignition/gazebo/components/Model.hh>
#include <ignition/gazebo/components/Name.hh>
#include <ignition/plugin/Register.hh>

namespace fire_robot {
namespace sim = ignition::gazebo;
namespace comp = sim::components;

class FilmCamera final : public sim::System, public sim::ISystemPreUpdate {
  sim::Entity robot{sim::kNullEntity}, camera{sim::kNullEntity};
  ignition::math::Pose3d filtered;
  bool initialized{false};

 public:
  void PreUpdate(const sim::UpdateInfo &info,
                 sim::EntityComponentManager &ecm) override {
    if (info.paused) return;
    if (robot == sim::kNullEntity)
      robot = ecm.EntityByComponents(comp::Model(), comp::Name("fire_robot"));
    if (camera == sim::kNullEntity)
      camera = ecm.EntityByComponents(comp::Model(),
                                     comp::Name("proof_perspective_camera"));
    if (robot == sim::kNullEntity || camera == sim::kNullEntity) return;
    const auto pose = sim::worldPose(robot, ecm);
    const auto yaw = pose.Rot().Yaw();
    const auto c = std::cos(yaw), s = std::sin(yaw);
    const auto &p = pose.Pos();
    const ignition::math::Vector3d eye(p.X()-1.0*c-.3*s,
                                       p.Y()-1.0*s+.3*c, 4.0);
    const ignition::math::Vector3d target(p.X()+.45*c, p.Y()+.45*s, .7);
    const auto delta = target-eye;
    ignition::math::Pose3d desired(eye, ignition::math::Quaterniond(
        0., -std::atan2(delta.Z(), std::hypot(delta.X(), delta.Y())),
        std::atan2(delta.Y(), delta.X())));
    const double dt = std::chrono::duration<double>(info.dt).count();
    const double alpha = 1.-std::exp(-std::max(0., dt)/.15);
    if (!initialized) { filtered = desired; initialized = true; }
    else {
      filtered.Pos() += alpha*(desired.Pos()-filtered.Pos());
      filtered.Rot() = ignition::math::Quaterniond::Slerp(
          alpha, filtered.Rot(), desired.Rot());
    }
    sim::Model(camera).SetWorldPoseCmd(ecm, filtered);
  }
};
}
IGNITION_ADD_PLUGIN(fire_robot::FilmCamera, ignition::gazebo::System,
                    fire_robot::FilmCamera::ISystemPreUpdate)
IGNITION_ADD_PLUGIN_ALIAS(fire_robot::FilmCamera, "fire_robot::FilmCamera")
