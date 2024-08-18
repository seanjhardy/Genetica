#include <functional>

class Animation {
public:
    Animation() = default;
    Animation(std::function<void(float)> setter, float fromValue = 0.0, float toValue = 1.0, float duration = 0.1, float delay = 0.0)
            : fromValue(fromValue), toValue(toValue), duration(duration),
            delay(delay), currentDelay(delay), setter(std::move(setter)) {}

    void update(float dt) {
        if (currentDelay > 0) {
            currentDelay -= dt;
            return;
        }
        if (progress <= 1) {
            progress += dt / duration;
            setter(fromValue * (1 - progress) + toValue * progress);
        } else {
            completed = true;
        }
    }

    void reset() {
        progress = 0;
        currentDelay = delay;
        completed = false;
    }

    bool completed;
    float duration = 0.5;
    float delay = 0;
    float currentDelay = 0;
    float progress = 0;
    float fromValue = 0;
    float toValue = 1;
    std::function<void(float)> setter;
};