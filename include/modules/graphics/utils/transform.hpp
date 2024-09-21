#ifndef TRANSFORM
#define TRANSFORM

class UITransform {
public:
    enum class TransformType {
        Scale,
    };

    float getValue() const {
        return m_currentValue;
    }

    float getDuration() const {
        return m_duration;
    }

    void setCurrentValue(float animationProgress) {
        m_currentValue = m_startValue * (1 - animationProgress) + m_value * animationProgress;
    }

    static UITransform Scale(float value, float duration=0.1f) { return {TransformType::Scale, value, 1, duration}; }

private:
    UITransform(TransformType type, float value, float startValue = 1.0f, float duration = 0.1f)
    : m_type(type), m_value(value), m_currentValue(value), m_startValue(startValue), m_duration(duration) {}
    TransformType m_type;
    float m_startValue;
    float m_value;
    float m_currentValue = 0;
    float m_duration;
};

#endif