#ifndef GENOME
#define GENOME

#include <map>
#include <string>
#include <vector>
#include <modules/graphics/vertexManager.hpp>

using namespace std;

class Genome {
public:
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

    map<int, string> hoxGenes;
    vector<int> hoxGeneOrder;

    sf::RenderTexture cachedTexture;
    //map<int, string> neurologicalGenes;

    Genome();
    Genome(const Genome& other) {
        hoxGenes = other.hoxGenes;
        hoxGeneOrder = other.hoxGeneOrder;
    }

    void init(Template templateType = Template::RANDOM);

    void render(VertexManager& vertexManager);
    static constexpr int HOX_SIZE = 100;

    void addHoxGene(int key, const string& value, int position=-1);
    void removeGene(int key);

    [[nodiscard]] bool contains(int key) const;
    [[nodiscard]] string at(int key) const;
    [[nodiscard]] map<int, string> getGenes() const;
    [[nodiscard]] string toString() const;
};

#endif //GENOME