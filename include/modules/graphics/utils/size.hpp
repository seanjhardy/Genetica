#ifndef SIZE_VALUE
#define SIZE_VALUE

class Size {
public:
    enum class Mode { Pixel, Percent, Flex };

    static Size Pixel(float value) { return {Mode::Pixel, value}; }
    static Size Percent(float value) { return {Mode::Percent, value}; }
    static Size Flex(float value = 1.0f) { return {Mode::Flex, value}; }

    [[nodiscard]] Mode getMode() const { return m_mode; }
    [[nodiscard]] float getValue() const { return m_value; }

private:
    Size(Mode mode, float value) : m_mode(mode), m_value(value) {}
    Mode m_mode;
    float m_value;
};

#endif