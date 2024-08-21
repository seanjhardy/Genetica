#include <simulator/simulator.hpp>

Entity::Entity(float2 pos) : pos(pos){
    entityID = Simulator::get().nextEntityID();
    setPos(pos);
};