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
        //TODO: Add more templates
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