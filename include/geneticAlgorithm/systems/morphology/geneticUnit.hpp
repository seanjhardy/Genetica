#ifndef GENETIC_ELEMENT
#define GENETIC_ELEMENT

#include <geneticAlgorithm/cellParts/cell.hpp>

/**
 * A genetic element that can be part of a gene regulatory network
 */
class GeneticUnit {
public:
    enum class Type {
        Promoter,
        Effector,
        Factor,
    } type;
    static constexpr float DISTANCE_THRESHOLD = 1.0f;
    static constexpr int N = 3;

    bool sign;
    bool active;
    float modifier;
    float embedding[3]{};

    GeneticUnit(Type type, bool active, bool sign,
                float modifier, const float* embedding)
        : type(type), active(active), sign(sign), modifier(modifier){
        for (int i = 0; i < N; i++) {
            this->embedding[i] = embedding[i];
        }
    }

    [[nodiscard]] float calculateAffinity(const GeneticUnit& other) const {
        float distance = 0.0;
        for (int i = 0; i < GeneticUnit::N; i++) {
            distance += std::pow(embedding[i] - other.embedding[i], 2);
        }
        distance = std::sqrt(distance);
        if (distance > GeneticUnit::DISTANCE_THRESHOLD) return 0.0;

        return (sign * other.sign) ? 1 : -1 *
             (2.0f * std::abs(modifier * other.modifier)
                * (GeneticUnit::DISTANCE_THRESHOLD - distance)) /
             (10.0f * distance + std::abs(modifier * other.modifier));
    }
};

class Effector : public GeneticUnit {
public:
    enum class EffectorType {
        //Actions
        Divide,
        Die,
        Freeze,

        //Parameters
        Radius,
        DaughterDistance,
        DivisionVectorLength,
        DivisionVectorAngle,
    } effectorType;

    Effector(EffectorType effectorType,
             bool active,
             bool sign,
             float modifier,
             const float* embedding)
        : GeneticUnit(Type::Effector, sign, active, modifier, embedding),
          effectorType(effectorType) {}

    void activate(Cell& cell, float activationLevel) {};
};

/**
 * A gene which can either be on or off
 */
class Gene : public GeneticUnit {
public:
    enum class FactorType {
        // External Factors
        ExternalFactorP1, // A morphogen from the environment
        ExternalFactorP2, // A morphogen from the environment
        ExternalFactorP3, // A morphogen from the environment
        Constant, // A value of 1
        Time, // A value that increases over time
        Generation, // The generation of a particular
        Congestion, // The number of nearby cells
        Energy, // The energy of the life form (maybe add energy per cell?)
        MaternalFactor, // A factor from the genome

        // Genetic factors
        InternalProduct,
        ExternalProduct,
        Receptor,
    } factorType{};
    virtual ~Gene() = default;

    Gene(FactorType factorType,
         bool active,
         bool sign,
         float modifier,
         const float* embedding)
        : GeneticUnit(Type::Factor, active, sign, modifier, embedding),
          factorType(factorType){}

    virtual void express(){};
    void interact(const std::unordered_map<Gene*, float> morphogenLevels) {};
};

/**
 * An element that promotes the level of other elements
 */
class Promoter : public GeneticUnit {
public:
    enum class PromoterType {
        Additive,
        Multiplicative
    } promoterType;

    virtual ~Promoter() = default;
    Promoter(PromoterType promoterType,
             bool active,
             bool sign,
             float modifier,
             const float* embedding)
        : GeneticUnit(Type::Promoter, active, sign, modifier, embedding),
          promoterType(promoterType) {}

    float calculateActivity(std::vector<Gene>& factors,
                            std::map<std::pair<Promoter*, Gene*>, float>& promoterFactorAffinities,
                            std::unordered_map<Gene*, float>& levels) {
        float activity = 0.0f;
        for (auto& factor : factors) {
            float affinity = promoterFactorAffinities.at({this, &factor});
            activity += levels[&factor] * affinity;
        }
        return activity;
    }
};

#endif