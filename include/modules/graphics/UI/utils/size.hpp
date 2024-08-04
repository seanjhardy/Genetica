#ifndef SIZE
#define SIZE

class Size {
public:
    enum class Mode { Pixel, Percent, Flex };

    static Size Pixel(float value) { return Size(Mode::Pixel, value); }
    static Size Percent(float value) { return Size(Mode::Percent, value); }
    static Size Flex(float value = 1.0f) { return Size(Mode::Flex, value); }

    Mode getMode() const { return m_mode; }
    float getValue() const { return m_value; }

private:
    Size(Mode mode, float value) : m_mode(mode), m_value(value) {}
    Mode m_mode;
    float m_value;
};

#endif