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

    void setCurrentValue(float animationProgress) {
        m_currentValue = m_startValue * (1 - animationProgress) + m_value * animationProgress;
    }

    static UITransform Scale(float value) { return {TransformType::Scale, value, 1}; }

private:
    UITransform(TransformType type, float value, float startValue = 1.0f)
    : m_type(type), m_value(value), m_currentValue(value), m_startValue(startValue) {}
    TransformType m_type;
    float m_startValue;
    float m_value;
    float m_currentValue = 0;
};

#endif