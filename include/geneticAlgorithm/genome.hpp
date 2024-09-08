#include <map>
#include <string>
#include <vector>

using namespace std;

class Genome {
public:
    static constexpr int HOX_SIZE = 85;

    map<int, string> hoxGenes;
    vector<int> hoxOrder;

    Genome();
    Genome(const Genome& other) = default;

    void addHoxGene(int key, const string& value, int position=-1);
    void removeGene(int key);

    [[nodiscard]] bool contains(int key) const;
    [[nodiscard]] string at(int key) const;
    [[nodiscard]] map<int, string> getGenes() const;
    [[nodiscard]] string toString() const;
};