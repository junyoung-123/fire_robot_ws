// Passive ECM reader. No Configure/PreUpdate callbacks or physical commands.
#include <chrono>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/EntityComponentManager.hh>
#include <ignition/gazebo/components/Collision.hh>
#include <ignition/gazebo/components/ContactSensor.hh>
#include <ignition/gazebo/components/ContactSensorData.hh>
#include <ignition/gazebo/components/Name.hh>
#include <ignition/gazebo/components/ParentEntity.hh>
#include <ignition/msgs/contacts.pb.h>
#include <ignition/plugin/Register.hh>
#include <ignition/transport/Node.hh>

namespace fire_robot {
namespace sim = ignition::gazebo;
namespace comp = sim::components;

class ContactAudit final : public sim::System, public sim::ISystemPostUpdate {
  struct Stream {
    std::vector<sim::Entity> collisions;
    ignition::transport::Node::Publisher publisher;
    ignition::msgs::Contacts pending;
    std::set<std::pair<uint64_t,uint64_t>> pairs;
  };
  ignition::transport::Node node;
  std::map<std::string,Stream> streams;
  std::chrono::steady_clock::duration lastPublish{};

 public:
  void PostUpdate(const sim::UpdateInfo &info,
                  const sim::EntityComponentManager &ecm) override {
    if (info.paused) return;
    if (streams.size() < 5) {
      ecm.Each<comp::ContactSensor,comp::ParentEntity>(
          [&](const sim::Entity &, const comp::ContactSensor *sensor,
              const comp::ParentEntity *parent) {
        auto sdf = sensor->Data();
        if (!sdf->HasElement("contact")) return true;
        auto contact = sdf->GetElement("contact");
        const auto topic = contact->Get<std::string>("topic", "").first;
        const std::set<std::string> allowed = {"lever","latch","panel","jamb","strike"};
        if (topic.rfind("/proof/contact/",0) != 0) return true;
        const auto part = topic.substr(std::string("/proof/contact/").size());
        if (!allowed.count(part) || streams.count(part)) return true;
        Stream stream;
        for (auto collision = contact->GetElement("collision"); collision;
             collision = collision->GetNextElement("collision")) {
          auto ids = ecm.ChildrenByComponents(parent->Data(),comp::Collision(),
                                              comp::Name(collision->Get<std::string>()));
          stream.collisions.insert(stream.collisions.end(),ids.begin(),ids.end());
        }
        if (!stream.collisions.empty()) {
          stream.publisher = node.Advertise<ignition::msgs::Contacts>("/proof/audit/"+part);
          streams.emplace(part,std::move(stream));
        }
        return true;
      });
    }
    const bool publish = info.simTime-lastPublish >= std::chrono::milliseconds(50);
    for (auto &[part,stream] : streams) {
      bool valid = true;
      for (const auto entity : stream.collisions) {
        const auto *data = ecm.Component<comp::ContactSensorData>(entity);
        if (!data) { valid = false; break; }
        // Aggregate collision pairs at every physics step; the 20 Hz message
        // retains even a contact shorter than one publication interval.
        for (const auto &contact : data->Data().contact()) {
          if (stream.pairs.emplace(contact.collision1().id(),contact.collision2().id()).second)
            stream.pending.add_contact()->CopyFrom(contact);
        }
      }
      if (publish && valid) {
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime).count();
        auto *stamp = stream.pending.mutable_header()->mutable_stamp();
        stamp->set_sec(ns/1000000000);
        stamp->set_nsec(ns%1000000000);
        stream.publisher.Publish(stream.pending); // Empty is a valid heartbeat.
        stream.pending.Clear();
        stream.pairs.clear();
      }
    }
    if (publish) lastPublish=info.simTime;
  }
};
}
IGNITION_ADD_PLUGIN(fire_robot::ContactAudit, ignition::gazebo::System,
                    ignition::gazebo::ISystemPostUpdate)
IGNITION_ADD_PLUGIN_ALIAS(fire_robot::ContactAudit,"fire_robot::ContactAudit")
