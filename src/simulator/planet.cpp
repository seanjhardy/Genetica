#include <simulator/planet.hpp>
#include <modules/graphics/shaderManager.hpp>
#include <modules/utils/print.hpp>
#include <simulator/simulator.hpp>

std::map<std::string, Planet> Planet::planets = {};
std::vector<std::string> Planet::planetNames = {};
int Planet::current = 0;

Planet::Planet(std::string name){
    this->name = name;
    thumbnail = name;
    thumbnail.erase(std::remove(thumbnail.begin(), thumbnail.end(), '-'), thumbnail.end());
    std::transform(thumbnail.begin(), thumbnail.end(), thumbnail.begin(), ::tolower);
}

void Planet::update() {
    if (Simulator::get().getRealTime() - lastUpdate > 0.02) {
        lastUpdate = Simulator::get().getRealTime();
        updateMap();
    }
}

void Planet::render(VertexManager &vertexManager) {
    vertexManager.addSprite(mapSprite);
}

void Planet::setBounds(sf::FloatRect bounds) {
    mapBounds = bounds;
    for (auto& n : noise) {
        n->update = true;
    }
    updateMap();
}

sf::FloatRect Planet::getBounds() {
    return mapBounds;
}

void Planet::reset() {
    mapSeed = Random::random(10000);
    for (auto& n : noise) {
        n->update = true;
    }
    updateMap();
}

void Planet::updateMap() {
    bool update = false;
    for (auto& n : noise) {
        if (n->update || n->base.animated) {
            update = true;
        }
    }
    if (!update) return;

    sf::Texture map;
    int x_size = round(mapBounds.width / MAP_SCALE);
    int y_size = round(mapBounds.height / MAP_SCALE);
    texture.create(x_size, y_size);
    texture.clear(sf::Color::Black);

    // Seed for randomness
    float seed = mapSeed;
    if (seed == -1) {
        seed = Random::random(10000);
    }

    // Draw main colours
    for (int i = 0; i < noise.size(); i++) {
        noise[i]->applyToTexture(&texture, {mapBounds.width, mapBounds.height}, seed + i);
    }

    texture.display();
    mapSprite = sf::Sprite(texture.getTexture());
    mapSprite.setPosition({0, 0});
    mapSprite.setScale(MAP_SCALE, MAP_SCALE);
}

Planet* Planet::getRandom() {
    std::string planet = planetNames[current];//planetNames[Random::random((int)planetNames.size())]
    if (current < planetNames.size() - 1) {
        current += 1;
    } else {
        current = 0;
    }
    return &planets.at(planet);
}

void Planet::init() {
    Planet mars("Mars");
    mars.temperature = -5.0f;
    mars.noise = {
      new Add(Noise({sf::Color(61, 40, 54),
             sf::Color(81, 50, 62),
             sf::Color(172, 47, 68)},
            5.0f, 0.7f, 0.7f, true)),
    };
    planets.insert({mars.name, mars});
    planetNames.push_back(mars.name);

    Planet cyrus("Cyrus");
    cyrus.temperature = 70.0f;
    cyrus.noise = {
      new Add(Noise(
            {sf::Color(43, 30, 39),
             sf::Color(61, 41, 54),
             sf::Color(82, 51, 63),
             sf::Color(143, 77, 87)},
            8.0f, 1.0f, 0.7f, true, true)),
      new Add(Noise(
            {sf::Color(0, 0, 0, 0),
             sf::Color(0, 0, 0, 0),
             sf::Color(0, 0, 0, 0),
             sf::Color(0, 0, 0, 0),
             sf::Color(0, 0, 0, 0),
             sf::Color(0, 0, 0, 0),
             sf::Color(0, 0, 0, 0),
             sf::Color(0, 0, 0, 0),
             sf::Color(173, 47, 69, 0),
             //sf::Color(173, 47, 69),
             sf::Color(230, 69, 57),
             sf::Color(255, 180, 51),
             sf::Color(230, 69, 57),
             //sf::Color(173, 47, 69),
             sf::Color(173, 47, 69, 0),
             sf::Color(0, 0, 0, 0),
             sf::Color(0, 0, 0, 0),
             sf::Color(0, 0, 0, 0),
             sf::Color(0, 0, 0, 0),
             sf::Color(0, 0, 0, 0),
             sf::Color(0, 0, 0, 0),
             sf::Color(0, 0, 0, 0),
             sf::Color(0, 0, 0, 0),},
            10.0f, 0.1f, 0.7f, true, true)),

    };
    planets.insert({cyrus.name, cyrus});
    planetNames.push_back(cyrus.name);

    Planet syth("Syth");
    syth.temperature = 70.0f;
    syth.noise = {
      new Add(Noise(
        {sf::Color(43, 30, 39),
         sf::Color(61, 41, 54),
         sf::Color(82, 51, 63),
         sf::Color(143, 77, 87)},
        6.0f, 1.0f, 0.0f, true)),
    };
    planets.insert({syth.name, syth});
    planetNames.push_back(syth.name);

    Planet xeria("Xeria");
    xeria.temperature = 20.0f;
    xeria.noise = {
      new Add(Noise(
        {sf::Color(8, 92, 110),
         sf::Color(52, 119, 134),
         sf::Color(62, 146, 165),
         sf::Color(79, 164, 184)},
        6.0f, 1.0f, 0.0f, true)),
    };
    planetNames.push_back(xeria.name);
    planets.insert({xeria.name, xeria});

    Planet xeriab("Xeria-b");
    xeriab.temperature = 20.0f;
    xeriab.noise = {
      new Add(Noise(
        {sf::Color(8, 92, 110),
         sf::Color(52, 119, 134),
         sf::Color(62, 146, 165),
         sf::Color(79, 164, 184)},
        6.0f, 1.0f, 0.0f, true)),
    };
    planetNames.push_back(xeriab.name);
    planets.insert({xeriab.name, xeriab});

    Planet ichigo("Ichigo");
    ichigo.temperature = 40.0f;
    ichigo.noise = {
      new Add(Noise(
            {sf::Color(11, 104, 128),
             sf::Color(27, 143, 171),},
            3.0f, 0.1f, 8.0f, true)),
    };
    planetNames.push_back(ichigo.name);
    planets.insert({ichigo.name, ichigo});

    Planet b12axo("B12-Axo");
    b12axo.temperature = -10.0f;
    b12axo.noise = {
      new Add(Noise(
            {sf::Color(42, 46, 73),
             sf::Color(58, 63, 94),
             sf::Color(76, 104, 133)},
            6.0f, 1.0f, 0.0f, true)),
    };
    planetNames.push_back(b12axo.name);
    planets.insert({b12axo.name, b12axo});

    Planet aridium("Aridium");
    aridium.temperature = -30.0f;
    aridium.noise = {
      new Add(Noise(
            {sf::Color(41, 49, 21),
             sf::Color(67, 83, 29),
             sf::Color(97, 95, 41),
             sf::Color(146, 117, 62),
             sf::Color(166, 123, 80)},
            6.0f, 1.0f, 0.0f, true)),
    };
    planetNames.push_back(aridium.name);
    planets.insert({aridium.name, aridium});

    Planet ocea("Ocea");
    ocea.temperature = 30.0f;
    ocea.noise = {
      new Add(Noise(
            {sf::Color(55, 68, 106),
             sf::Color(67, 81, 125),
             sf::Color(31, 100, 156),
             sf::Color(79, 164, 184),
             sf::Color(146, 117, 62)},
            6.0f, 1.0f, 0.0f, true)),
    };
    planetNames.push_back(ocea.name);
    planets.insert({ocea.name, ocea});

    Planet gundam("Gundam");
    gundam.temperature = 15.0f;
    gundam.noise = {
      new Add(Noise(
            {sf::Color(51, 34, 25),
             sf::Color(61, 46, 38),
             sf::Color(80, 51, 49),
             sf::Color(92, 59, 57)},
            6.0f, 1.0f, 0.2f, true)),
      new Add(Noise(
          {sf::Color(0, 0, 0, 0),
          sf::Color(0, 0, 0, 0),
          sf::Color(43, 118, 92, 0),
          //sf::Color(22, 57, 59),
          sf::Color(43, 118, 92, 255),
          sf::Color(65, 177, 94, 255)},
            6.0f, 1.0f, 0.2f, true
          ))
    };
    planetNames.push_back(gundam.name);
    planets.insert({gundam.name, gundam});

    Planet roche("Roche");
    roche.temperature = 70.0f;
    roche.noise = {
      new Add(Noise(
            {sf::Color(59, 32, 39),
             sf::Color(59, 32, 39),
             sf::Color(171, 81, 48),
             sf::Color(240, 181, 65)},
            10.0f, 1.0f, 0.0f, true)),
    };
    planetNames.push_back(roche.name);
    planets.insert({roche.name, roche});

    Planet xia("Xia");
    xia.temperature = -25.0f;
    xia.noise = {
      new Add(Noise(
            {sf::Color(56, 39, 9),
             sf::Color(112, 50, 19),
             sf::Color(64, 151, 144),
             sf::Color(96, 227, 164),
             sf::Color(112, 50, 19),
             sf::Color(168, 33, 28)},
            8.0f, 0.2f, 0.0f, true)),
    };
    planetNames.push_back(xia.name);
    planets.insert({xia.name, xia});

    Planet glau("Glau");
    glau.temperature = 25.0f;
    glau.noise = {
      new Add(Noise(
        {sf::Color(11, 10, 14),
         sf::Color(16, 16, 23),
         sf::Color(25, 29, 34)},
        6.0f, 1.0f, 0.0f, true)),
    };
    planetNames.push_back(glau.name);
    planets.insert({glau.name, glau});

    Planet eden("Eden");
    eden.temperature = 25.0f;
    eden.noise = {
      new Add(Noise(
            {sf::Color(26, 72, 59),
             sf::Color(26, 90, 45),
             sf::Color(38, 134, 39)},
            6.0f, 0.3f, 3.0f, true)),
    };
    planetNames.push_back(eden.name);
    planets.insert({eden.name, eden});

    Planet zolo("Zolo");
    zolo.temperature = 5.0f;
    zolo.noise = {
      new Add(Noise(
            {sf::Color(70, 42, 18),
             sf::Color(120, 74, 32)},
            6.0f, 1.0f, 0.1f, true)),
      new Add(Noise(
            {sf::Color(0, 0, 0, 0),
             sf::Color(0, 0, 0, 0),
             sf::Color(0, 0, 0, 0),
             sf::Color(52, 59, 22, 0),
             sf::Color(52, 59, 22),
             sf::Color(104, 138, 42)},
            8.0f, 0.2f, 0.1f, true)),
    };
    planetNames.push_back(zolo.name);
    planets.insert({zolo.name, zolo});

    Planet riula("Riula");
    riula.temperature = -15.0f;
    riula.noise = {
      new Add(Noise(
            {sf::Color(4, 25, 29),
             sf::Color(11, 50, 58),},
            6.0f, 1.0f, 0.0f, true)),
    };
    planetNames.push_back(riula.name);
    planets.insert({riula.name, riula});

    Planet flax("Flax");
    flax.temperature = -35.0f;
    flax.noise = {
      new Add(Noise(
        {sf::Color(35, 15, 3),
         sf::Color(70, 42, 18),
         sf::Color(120, 74, 32)},
        6.0f, 0.2f, 0.0f, true)),
    };
    planetNames.push_back(flax.name);
    planets.insert({flax.name, flax});

    Planet romea("Romea");
    romea.temperature = 20.0f;
    romea.noise = {
      new Add(Noise(
            {sf::Color(22, 62, 83),
             sf::Color(34, 87, 103),
             sf::Color(45, 118, 140),
             sf::Color(51, 155, 148)},
            6.0f, 0.3f, 0.0f, true)),
        new Mask(Noise(
            {sf::Color(0,0,0, 0),
             sf::Color(0,0,0, 0),
             sf::Color(0,0,0, 0),
             sf::Color(10, 68, 22, 0),
             sf::Color(10, 68, 22, 150),
             sf::Color(41, 136, 20, 150)},
            5.0f, 0.1f, 0.0f, true
          ),
        Noise(
          {sf::Color(0, 0, 0, 0),
           sf::Color(0, 0, 0, 0),
           sf::Color(0, 0, 0, 0),
           sf::Color(0, 0, 0, 0),
           sf::Color(0, 0, 0, 0),
           sf::Color(0, 0, 0, 0),
           sf::Color(0, 0, 0, 255),
           sf::Color(0, 0, 0, 0),
           sf::Color(0, 0, 0, 0),
           sf::Color(0, 0, 0, 0),
           sf::Color(0, 0, 0, 0),
           sf::Color(0, 0, 0, 0),
           sf::Color(0, 0, 0, 0),},
          10.0f, 0.2f, 2.0f, true)),
    };
    planetNames.push_back(romea.name);
    planets.insert({romea.name, romea});

    Planet emulo("Emulo");
    emulo.temperature = 10.0f;
    emulo.noise = {
        new Add(Noise(
            {sf::Color(35, 15, 3),
             sf::Color(70, 42, 18),
             sf::Color(120, 74, 32)},
            6.0f, 1.0f, 0.0f, true)),
        new Add(Noise(
          {sf::Color(0,0,0,0),
           sf::Color(104, 138, 42,0),
           sf::Color(104, 138, 42),
           sf::Color(169, 173, 55)},
            6.0f, 1.0f, 0.1f, true)),
    };
    planetNames.push_back(emulo.name);
    planets.insert({emulo.name, emulo});

    Planet azeuros("Azeuros");
    azeuros.temperature = -20.0f;
    azeuros.noise = {
      new Add(Noise(
            {sf::Color(46, 102, 126),
             sf::Color(69, 189, 188)},
            6.0f, 1.0f, 0.0f, true)),
    };
    planetNames.push_back(azeuros.name);
    planets.insert({azeuros.name, azeuros});

    Planet iridium("Iridium");
    iridium.temperature = 30.0f;
    iridium.noise = {
      new Add(Noise(
            {sf::Color(65, 43, 81),
             sf::Color(80, 64, 121)},
            6.0f, 1.0f, 0.0f, true)),
    };
    planetNames.push_back(iridium.name);
    planets.insert({iridium.name, iridium});

    Planet nauca("Nauca");
    nauca.temperature = 20.0f;
    nauca.noise = {
      new Add(Noise(
            {sf::Color(38, 21, 40),
             sf::Color(65, 43, 81),
             sf::Color(80, 64, 121)},
            6.0f, 0.4f, 0.0f, true)),
    };
    planetNames.push_back(nauca.name);
    planets.insert({nauca.name, nauca});

    Planet abogee("Abogee");
    abogee.temperature = 60.0f;
    abogee.noise = {
      new Add(Noise(
            {sf::Color(22, 57, 40),
             sf::Color(75, 171, 66)},
            6.0f, 1.0f, 0.0f, true)),
    };
    planetNames.push_back(abogee.name);
    planets.insert({abogee.name, abogee});

    Planet delune("Delune");
    delune.temperature = 25.0f;
    delune.noise = {
      new Add(Noise(
            {sf::Color(3, 2, 47),
             sf::Color(16, 10, 97),
             sf::Color(35, 22, 144)},
            6.0f, 0.1f, 0.0f, true)),
    };
    planetNames.push_back(delune.name);
    planets.insert({delune.name, delune});

    Planet cerebrus("Cerebrus");
    cerebrus.temperature = 0.0f;
    cerebrus.noise = {
      new Add(Noise(
            {sf::Color(137, 36, 82),
             sf::Color(62, 13, 35)},
            6.0f, 0.1f, 0.0f, true)),
    };
    planetNames.push_back(cerebrus.name);
    planets.insert({cerebrus.name, cerebrus});
}
