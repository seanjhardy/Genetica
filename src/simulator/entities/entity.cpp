#include <simulator/simulator.hpp>

Entity::Entity(float2 pos) : pos(pos){
    entityID = Simulator::get().getEnv().nextEntityID();
    Simulator::get().getEnv().addEntity(entityID, this);
    setPos(pos);
};