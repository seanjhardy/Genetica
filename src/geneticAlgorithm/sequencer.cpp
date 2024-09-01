#include "unordered_map"
#include "geneticAlgorithm/sequencer.hpp"
#include <geneticAlgorithm/cellParts/segmentType.hpp>
#include "modules/utils/genomeUtils.hpp"
#include <modules/utils/print.hpp>
#include <geneticAlgorithm/geneticAlgorithm.hpp>
#include <simulator/simulator.hpp>
#include <memory>
#include <string>
#include <utility>

void sequence(LifeForm* lifeForm, const std::map<int, string>& genome) {
    // Get the lowest index genome as the morphogen chromosome
    int lowestChromosomeIndex = std::numeric_limits<int>::max(); // Initialize to maximum integer value

    for (const auto& [key, _] : lifeForm->getGenome()) {
        if (key < lowestChromosomeIndex) {
            lowestChromosomeIndex = key;
        }
    }

    string header = genome.at(lowestChromosomeIndex);
    sequenceMorphogens(header);

    // Read genes
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
        if (key == lowestChromosomeIndex) continue;
        if (lifeForm->cellParts.at(key)->type == CellPartType::Type::SEGMENT) {
            auto* headCellPartData = new CellPartSchematic(lifeForm->cellParts.at(key).get(),
                                               false, 100, 0, 0);
            CellPartInstance* segment = new SegmentInstance(lifeForm, headCellPartData, nullptr);
            lifeForm->addCellPartInstance(segment);
            lifeForm->head = dynamic_cast<SegmentInstance *>(segment);
            lifeForm->head->centered = true;
            break;
        }
    }
}

MorphogenSystem sequenceMorphogens(string chromosome) {
    MorphogenSystem morphogenSystem = MorphogenSystem();
    while (chromosome.length() >= LifeForm::MORPHOGEN_DATA_SIZE) {
        //int type = 0;
        //    float2 pos;
        //    float startConcentration, endConcentration;
        //    float3 extra;
        float morphogenCode = readUniqueBaseRange(chromosome, 5);
        int type = readUniqueBaseRange(chromosome, 2)*3;
        float angle = readBaseRange(chromosome, 4) * (float)M_PI*2;

        std::unordered_map<int, float> interactions;
        morphogenSystem.addMorphogen(morphogenCode,
                                     {type, angle, 0, 0, {0,0,0}},
                                     interactions);
    }
    return morphogenSystem;
}

std::shared_ptr<CellPartType> sequenceChromosome(int key, string chromosome) {
    std::string rna = std::move(chromosome);
    bool isSegment = true;//readBase(chromosome, 0) > 0 || key == 1;

    if (isSegment) {
        float partCode = readBaseRange(rna, 15);
        unique_ptr<SegmentType> cellPart = std::make_unique<SegmentType>(partCode);

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
    readBaseRange(chromosome, LifeForm::HEADER_SIZE); // Skip header

    while (chromosome.length() >= LifeForm::CELL_DATA_SIZE) {
        float partCode = readBaseRange(chromosome, 15);

        // Get part with the most similar partCode (differentiable)
        int partID = -1;
        float minDifference = std::numeric_limits<float>::max();
        for (auto& [key, value]: lifeForm->cellParts) {
            float difference = std::abs(value->partCode - partCode);
            if (difference < minDifference) {
                minDifference = difference;
                partID = key;
            }
        }
        int buildPriority = int(readExpBaseRange(chromosome, 5) * 100);
        float angleOnBody = readBaseRange(chromosome, 4) * (float)M_PI;
        float angleFromBody = (readBaseRange(chromosome, 4) * 90.0f - 45.0f) * (float)M_PI / 180.0f;

        // Ensure that the partID exists and that the partToBuildFrom is a segment
        if (partToBuildFrom->type != CellPartType::Type::SEGMENT) return;

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

std::map<int, string> plantGenome() {
    std::map<int, string> genome;
    genome.insert({Simulator::get().getGA().nextGeneID(),
                    "2" // Symmetry
                    //"0" // Asexual
                    "22222" // Size
                    "001" // Growth energy
                    "222" // Growth rate
                    "222" // Child energy
                    "222" // Regeneration fraction
    });
    int head =Simulator::get().getGA().nextGeneID();
    genome.insert({head,
                   //PartCode
                   "000000000000000" // Head
                   "1111"//R
                   "2333"//G
                   "1113"//B
                    "222"//Start width
                    "222"//End width
                    "111"//Length
                    "0"//Bone
                    "0"//Muscle
                    "0"//Nerve
                    "3"//Fat
                    "000"//Bone density
                    "000"//Muscle strength
                    "222"//Fat size
                    // CELL PARTS
                    "111111111111111"//Part ID
                    "222222"//Build priority
                    "1122"//Angle on body
                    "1122"//Angle from body
                  });
    int body = Simulator::get().getGA().nextGeneID();
    genome.insert({body,
                    //PartCode
                   "111111111111111" // Head
                   "0111"//R
                   "3333"//G
                   "0111"//B
                   "222"//Start width
                   "122"//End width
                   "333"//Length
                   "0"//Bone
                   "0"//Muscle
                   "0"//Nerve
                   "3"//Fat
                   "000"//Bone density
                   "000"//Muscle strength
                   "222"//Fat size
                   // CELL PARTS
                   "111111111111111"//Part ID
                   "111111"//Build priority
                   "0022"//Angle on body
                   "1122"//Angle from body
                   "111111111111111"//Part ID
                   "111111"//Build priority
                   "3322"//Angle on body
                   "1122"//Angle from body
                  });
    return genome;
}