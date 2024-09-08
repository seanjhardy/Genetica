#include <simulator/simulator.hpp>
#include <simulator/entities/entity.hpp>

Entity::Entity(float2 pos) {
    entityID = Simulator::get().getEnv().nextEntityID();
    Simulator::get().getEnv().addEntity(entityID, this);
    setPos(pos);
};