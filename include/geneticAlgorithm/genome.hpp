#ifndef GENOME
#define GENOME

#include <map>
#include <string>
#include <vector>
#include <modules/graphics/vertexManager.hpp>

using namespace std;

class Genome {
public:
    static constexpr int HOX_SIZE = 100;
    enum class Template {
        RANDOM, // Completely random genome

        // PROKARYOTES
        PROKARYOTE, // Simple single-celled organisms that absorb nutrients from the environment

        //EUKARYOTIC PLANTS
        RANDOM_PLANT, // Multi-celled, complex organisms that photosynthesize to produce energy

        // EUKARYOTIC ANIMALS
        RANDOM_ANIMAL, // Multi-celled, complex organisms that move and consume other organic matter
        FLATWORM, // Simple creatures with behavior mediated by chemicals
        FISH, // Multi-celled, complex organisms with advanced nervous systems
        AMPHIBIAN, // Multi-celled, complex organisms that undergo metamorphosis
    };

    map<size_t, string> hoxGenes;
    vector<size_t> hoxGeneOrder;

    //sf::RenderTexture cachedTexture;

    Genome();
    Genome(const Genome& other) {
        hoxGenes = other.hoxGenes;
        hoxGeneOrder = other.hoxGeneOrder;
    }

    void init(Template templateType = Template::RANDOM);

    void render(VertexManager& vertexManager);

    void addHoxGene(size_t key, const string& value, int position=-1);
    void removeGene(size_t key);

    [[nodiscard]] bool contains(size_t key) const;
    [[nodiscard]] string at(size_t key) const;
    [[nodiscard]] map<size_t, string> getGenes() const;
    [[nodiscard]] string toString() const;
};


#endif //GENOME