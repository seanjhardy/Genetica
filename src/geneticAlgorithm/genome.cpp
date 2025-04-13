#include <geneticAlgorithm/genome.hpp>
#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include <modules/utils/random.hpp>
#include <simulator/simulator.hpp>

Genome::Genome() {}

bool Genome::contains(size_t key) const {
    return hoxGenes.contains(key);
}

string Genome::at(size_t key) const {
    return hoxGenes.at(key);
}

map<size_t, string> Genome::getGenes() const {
    return hoxGenes;
}

void Genome::addHoxGene(size_t key, const string& value, int position) {
    hoxGenes.insert({key, value});
    if (position == -1) {
        position = hoxGeneOrder.size();
    }
    hoxGeneOrder.insert(hoxGeneOrder.begin() + position, key);
}

void Genome::removeGene(size_t key) {
    hoxGenes.erase(key);
    hoxGeneOrder.erase(std::remove(hoxGeneOrder.begin(), hoxGeneOrder.end(), key), hoxGeneOrder.end());
}

string Genome::toString() const {
    string genomeString;
    for (auto& [key, value]: hoxGenes) {
        genomeString += value;
    }
    return genomeString;
}

void Genome::render(VertexManager& vertexManager) {
    /*auto cachedTexture = sf::RenderTexture();
    cachedTexture.create(400, 400);
    //cachedTexture.clear(sf::Color::Transparent);
    sf::RenderStates states;
    states.texture = &cachedTexture.getTexture();
    states.shader = ShaderManager::get("genome");
    states.blendMode = sf::BlendAdd;
    ShaderManager::get("genome")->setUniform("texture", sf::Shader::CurrentTexture);
    ShaderManager::get("genome")->setUniform("resolution", sf::Glsl::Vec2(400.0f, 400.0f));
    ShaderManager::get("genome")->setUniform("time", Simulator::get().getRealTime());

    // Draw genome background
    sf::RectangleShape rectangle(sf::Vector2f(400, 400));
    rectangle.setTextureRect(sf::IntRect(0, 0, 1, 1));
    rectangle.setFillColor(sf::Color::Transparent);
    cachedTexture.draw(rectangle, states);
    cachedTexture.display();
    vertexManager.addSprite(sf::Sprite(cachedTexture.getTexture()));*/

    // Draw DNA data
    sf::RectangleShape dna(sf::Vector2f(2, 2));
    float width = 400.0f;
    float height = 400.0f;
    int numBases = 0;
    for (auto& geneID : hoxGeneOrder) {
        numBases += hoxGenes.at(geneID).size();
    }
    float baseSize = sqrt((width * height) / numBases);
    float basesPerRow = width / baseSize;
    float yPos = 2;
    int baseIndex = 0;
    sf::Color backboneColour = sf::Color(255, 0, 0, 200);
    for (auto& geneID : hoxGeneOrder) {
        auto& gene = hoxGenes.at(geneID);
        for (char c: gene) {
            int base = int(c - '0');
            sf::Color color = sf::Color(0, 60, 0, 200);
            if (base == 1) {
                color = sf::Color(0, 120, 0, 200);
            } else if (base == 2) {
                color = sf::Color(0, 180, 0, 200);
            } else if (base == 3) {
                color = sf::Color(0, 255, 0, 200);
            }
            float x = float(baseIndex % int(basesPerRow)) * baseSize;
            if (baseIndex % int(basesPerRow) == 0) {
                yPos += baseSize;
            }
            vertexManager.addFloatRect({x, yPos+1, baseSize, baseSize}, color);
            baseIndex += 1;
        }
    }
}

void Genome::init(Template templateType) {
    if (templateType == Template::RANDOM) {
        int numHoxGenes = (int) Random::random(20, 100);
        int geneLength = 100;
        for (int i = 0; i < numHoxGenes; i++) {
            string randomGene;
            for (int j = 0; j < geneLength; j++) {
                randomGene += Random::randomBase();
            }
            int index = Simulator::get().getEnv().getGA().nextGeneID();
            hoxGenes.insert({index, randomGene});
            hoxGeneOrder.push_back(index);
        }
    }
}


