#include "unordered_map"
#include <geneticAlgorithm/environments/hyperLife/sequencer.hpp>
#include <geneticAlgorithm/environments/hyperLife/cellParts/segmentType.hpp>
#include <geneticAlgorithm/genomeUtils.hpp>
#include <memory>
#include <string>
#include <utility>

void sequence(LifeForm* lifeForm, const std::unordered_map<int, string>& genome) {
    // Get the lowest index genome as the header
    int lowestChromosomeIndex = std::numeric_limits<int>::max(); // Initialize to maximum integer value

    for (const auto& [key, _] : lifeForm->getGenome()) {
        if (key < lowestChromosomeIndex) {
            lowestChromosomeIndex = key;
        }
    }
    string header = genome.at(lowestChromosomeIndex);

    // Set meta variables from header
    lifeForm->symmetryType = std::vector<LifeForm::SymmetryType>{
        LifeForm::SymmetryType::NONE,
         LifeForm::SymmetryType::LOCAL,
         LifeForm::SymmetryType::GLOBAL,
         LifeForm::SymmetryType::RADIAL}[readBase(header)];

    lifeForm->reproductionType = readBase(header) == 3
            ? LifeForm::ReproductionType::SEXUAL
            : LifeForm::ReproductionType::ASEXUAL;
    lifeForm->size = max(readExpBaseRange(header, 5) * 5, 0.5f);
    lifeForm->growthEnergy = max(readBaseRange(header, 3) * 100.0f, 5.0f);
    lifeForm->growthRate = max(readBaseRange(header, 3), 0.1f);
    lifeForm->childEnergy = max(readBaseRange(header, 3) * 0.9f, 0.05f);
    lifeForm->regenerationFraction = min(readBaseRange(header, 3), 0.9f);

    // Read segment genomes
    for (auto& [key, value]: genome) {
        if (key == lowestChromosomeIndex) continue;
        auto cellPart = sequenceChromosome(key, value);
        lifeForm->cellParts[key] = std::move(cellPart);
    }

    // Construct body plan
    for (auto& [key, value]: genome) {
        if (key == lowestChromosomeIndex) continue;
        construct(lifeForm, key, value);
    }

    // Designate head part
    for (auto& [key, value]: genome) {
        if (key == lowestChromosomeIndex || lifeForm->head != nullptr) continue;
        if (lifeForm->cellParts.at(key)->type == CellPartType::Type::SEGMENT) {
            CellPartSchematic* headCellPartData = new CellPartSchematic(lifeForm->cellParts.at(key).get(),
                                               false, 100, 0, 0);
            CellPartInstance* segment = new SegmentInstance(lifeForm, headCellPartData, nullptr);
            lifeForm->addCellPartInstance(segment);
            lifeForm->head = dynamic_cast<SegmentInstance *>(segment);
            lifeForm->head->centered = true;
            break;
        }
    }
}

std::shared_ptr<CellPartType> sequenceChromosome(int key, string chromosome) {
    std::string rna = std::move(chromosome);
    bool isSegment = true;//readBase(chromosome, 0) > 0 || key == 1;

    if (isSegment) {
        unique_ptr<SegmentType> cellPart = std::make_unique<SegmentType>(key);
        int R = (int) (readBaseRange(rna, 4) * 255);
        int G = (int) (readBaseRange(rna, 4) * 255);
        int B = (int) (readBaseRange(rna, 4) * 255);
        cellPart->color = sf::Color(R, G, B);
        cellPart->startWidth = max(readExpBaseRange(rna, 3) * 20.0f, 0.05f);
        cellPart->endWidth = max(readExpBaseRange(rna, 3) * 20.0f, 0.05f);
        cellPart->length = max(readExpBaseRange(rna, 3) * 40.0f, 0.05f);
        cellPart->bone = readBase(rna) > 1;
        cellPart->muscle = readBase(rna) > 1;
        cellPart->nerve = readBase(rna) > 1;
        cellPart->fat = readBase(rna) > 1;
        cellPart->boneDensity = readBaseRange(rna, 3);
        cellPart->muscleStrength = readBaseRange(rna, 3);
        cellPart->fatSize = readBaseRange(rna, 3);
        return cellPart;
    } else {
        //TODO: Implement Protein synthesis
    }
    return std::unique_ptr<SegmentType>(new SegmentType(0));
}

// Generate a build schematic for children of the cell part
void construct(LifeForm* lifeForm, int key, string chromosome) {
    if (!lifeForm->cellParts.contains(key)) return;

    CellPartType* partToBuildFrom = lifeForm->cellParts.at(key).get();

    // Keep track of whether the first valid segment has been found (for ensuring radial symmetry around head only)
    bool headSegmentFound = false;

    while (chromosome.length() > LifeForm::CELL_DATA_SIZE) {
        int partID = int(readUniqueBaseRange(chromosome, 3) * 64);
        int buildPriority = int(readExpBaseRange(chromosome, 5) * 100);
        float angleOnBody = readUniqueBaseRange(chromosome, 2) * (float)M_PI * 2.0f;
        float angleFromBody = (readBaseRange(chromosome, 2) * 90.0f - 45.0f) * (float)M_PI / 180.0f;

        // Ensure that the partID exists and that the partToBuildFrom is a segment
        if (!lifeForm->cellParts.contains(partID) ||
            partToBuildFrom->type != CellPartType::Type::SEGMENT) return;

        // Skip adding CellPart if another CellPart with the same angle already exists
        for (CellPartSchematic child : dynamic_cast<SegmentType*>(partToBuildFrom)->children) {
            if (child.angleOnBody == angleOnBody) {
                continue;
            }
        }

        CellPartType* partToAdd = lifeForm->cellParts.at(partID).get();
        CellPartSchematic cellPartData(partToAdd, false, buildPriority, angleOnBody, angleFromBody);
        dynamic_cast<SegmentType*>(partToBuildFrom)->addChild(cellPartData);

        // If the symmetry partType is radial, and the part to build from is a segment,
        // and this is the first segment found (aka the head) then add a radial segment
        if (lifeForm->symmetryType == LifeForm::SymmetryType::RADIAL &&
            partToBuildFrom->type == CellPartType::Type::SEGMENT && !headSegmentFound) {
            headSegmentFound = true;
            CellPartSchematic cellPartDataRadial(partToAdd, true, buildPriority,
                                      angleOnBody + (float)M_PI, angleFromBody);
            dynamic_cast<SegmentType*>(partToBuildFrom)->addChild(cellPartDataRadial);
        }

        // Ignore radial symmetry for nodes at 0 and 180 degrees, pointing at 0 degrees (aka along the symmetry line)
        if ((angleOnBody == (float)M_PI || angleOnBody == 0) && angleFromBody == 0) {
            continue;
        }

        // If the symmetry partType is global or local, add a flipped segment
        // Local symmetry means every segment builds mirrored children,
        // Global symmetry means there's only one line of symmetry down the main axis
        // (Flipped parts are ignored during building for local symmetry not along the main axis)
        if (lifeForm->symmetryType == LifeForm::SymmetryType::GLOBAL ||
            lifeForm->symmetryType == LifeForm::SymmetryType::LOCAL) {
            CellPartSchematic cellPartDataFlipped(partToAdd, true, buildPriority,
                                             M_PI * 2 - angleOnBody, -angleFromBody);
            dynamic_cast<SegmentType*>(partToBuildFrom)->addChild(cellPartDataFlipped);
        }
    }
}