"""
wildcard_suite_gemma4.py — LoRa-Daddy True Wildcard Engine v2.0
================================================================
A REAL wildcard. Not just NSFW. Everything.

Content detection reads the user's instruction first:
  - If it smells NSFW → NSFW pools (original behaviour, expanded)
  - If it's clearly SFW → SFW pools (massive: vehicles, sports, fantasy, sci-fi,
    animation, nature, food, travel, animals, horror, historical, music, etc.)
  - If it's blank or totally ambiguous → full chaos mode, anything goes

Input anchoring: the user's prompt steers which sub-pools fire.
  "man racing a car"  → car type, track, era, weather, POV randomised
  "woman in a dress"  → destination, occasion, mood, time of day randomised
  "spongebob"         → animation world locked, adventure scenario randomised
  Blank               → complete coin-flip across everything

All content is still fed to the LLM as an instruction block — the LLM
writes the final prompt. This file just gives it an insanely rich starting point.
"""

import random
import re

# ══════════════════════════════════════════════════════════════════════════════
#  DETECTION — read user input, decide which universe to build from
# ══════════════════════════════════════════════════════════════════════════════

_NSFW_SIGNALS = [
    "sex", "fuck", "naked", "nude", "cock", "pussy", "breast", "penis",
    "penetrat", "oral", "bdsm", "strip", "lap dance", "blowjob", "handjob",
    "cum", "orgasm", "masturbat", "dildo", "vibrator", "erotic", "porn",
    "horny", "slutt", "whore", "lingerie", "topless", "ass", "tits",
    "nipple", "clit", "vagina", "anal", "fetish", "bondage", "dominat",
    "submiss", "hentai", "lewd", "nsfw",
]

_VEHICLE_SIGNALS = [
    "car", "race", "racing", "drive", "driving", "truck", "motorcycle",
    "motorbike", "bike", "ferrari", "lamborghini", "formula", "f1", "nascar",
    "drift", "rally", "supercar", "van", "bus", "train", "plane", "jet",
    "helicopter", "ship", "boat", "yacht", "submarine", "tank",
]

_SPORTS_SIGNALS = [
    "sport", "football", "soccer", "basketball", "tennis", "boxing",
    "fight", "wrestling", "mma", "swim", "swimming", "running", "marathon",
    "cycling", "skateboard", "surf", "ski", "snowboard", "golf", "baseball",
    "cricket", "rugby", "volleyball", "gymnastics", "parkour", "climbing",
]

_ANIMATION_SIGNALS = [
    "anime", "cartoon", "animated", "spongebob", "patrick", "goku",
    "naruto", "pikachu", "pokemon", "disney", "pixar", "simpsons",
    "looney", "tom and jerry", "rick and morty", "adventure time",
    "gravity falls", "studio ghibli", "miyazaki", "sonic", "mario",
    "zelda", "link", "batman", "superman", "spider-man", "avengers",
    "mickey mouse", "bugs bunny", "homer simpson",
]

_FANTASY_SIGNALS = [
    "dragon", "wizard", "witch", "magic", "fantasy", "elf", "dwarf",
    "orc", "dungeon", "sword", "knight", "king", "queen", "castle",
    "medieval", "myth", "legend", "quest", "hero", "paladin", "mage",
    "warlock", "sorcerer", "enchanted", "goblin", "troll", "fairy",
]

_SCIFI_SIGNALS = [
    "space", "alien", "robot", "cyberpunk", "future", "sci-fi", "scifi",
    "spaceship", "galaxy", "planet", "astronaut", "android", "cyborg",
    "laser", "warp", "starship", "mars", "moon", "nasa", "dystopia",
    "post-apocalyptic", "mech", "exosuit", "starwars", "star wars", "star trek",
]

_ANIMAL_SIGNALS = [
    "dog", "cat", "horse", "lion", "tiger", "elephant", "bear",
    "wolf", "fox", "rabbit", "bird", "eagle", "owl", "dolphin",
    "whale", "shark", "snake", "crocodile", "gorilla", "monkey",
    "panda", "penguin", "koala", "kangaroo", "deer", "giraffe",
    "zebra", "cheetah", "leopard",
]

_FOOD_SIGNALS = [
    "cook", "cooking", "chef", "kitchen", "food", "bake", "baking",
    "restaurant", "cafe", "coffee", "pizza", "ramen", "sushi", "burger",
    "pasta", "bread", "cake", "dessert", "recipe",
]

_HORROR_SIGNALS = [
    "horror", "zombie", "vampire", "ghost", "haunted", "monster",
    "demon", "witch", "curse", "evil", "creepy", "scary",
    "nightmare", "slasher", "thriller", "apocalypse",
]

_NATURE_SIGNALS = [
    "mountain", "forest", "jungle", "ocean", "beach", "river", "lake",
    "desert", "volcano", "glacier", "waterfall", "cave", "island",
    "nature", "wilderness", "camping", "hiking", "exploring",
]

_MUSIC_SIGNALS = [
    "music", "band", "concert", "guitar", "piano", "drum", "sing",
    "singer", "rapper", "dj", "festival", "stage", "performance",
    "rock", "jazz", "classical", "hip hop", "pop star",
]

_HISTORICAL_SIGNALS = [
    "ancient", "roman", "greek", "egypt", "viking", "samurai", "ninja",
    "pirate", "cowboy", "western", "war", "wwii", "ww2", "civil war",
    "napoleonic", "renaissance", "victorian", "medieval", "pharaoh",
]

_DIALOGUE_SIGNALS = [
    "dialogue", "talking", "conversation", "speech", "argue", "argument",
    "shout", "whisper", "yell", "monologue", "quote", "saying", "words",
    "speak", "tell me", "what they say", "lines", "script",
]

_CRIME_SIGNALS = [
    "heist", "robbery", "crime", "criminal", "detective", "murder",
    "noir", "gangster", "mafia", "mob", "hitman", "assassin",
    "cop", "police", "fbi", "interrogation", "chase", "fugitive",
    "smuggler", "cartel", "prison", "escape",
]

_WEATHER_SIGNALS = [
    "storm", "tornado", "hurricane", "blizzard", "thunderstorm",
    "lightning", "flood", "earthquake", "tsunami", "avalanche",
    "wildfire", "drought", "fog", "hail", "cyclone",
]

_URBAN_SIGNALS = [
    "city", "street", "urban", "skyscraper", "downtown", "subway",
    "graffiti", "alley", "rooftop", "neighbourhood", "market", "crowd",
    "nightclub", "bar", "pub", "neon", "billboard",
]


def _detect_content_type(instruction: str) -> str:
    if not instruction or not instruction.strip():
        return "chaos"
    low = instruction.lower()
    if any(s in low for s in _NSFW_SIGNALS):
        return "nsfw"
    if any(s in low for s in _ANIMATION_SIGNALS):
        return "animation"
    if any(s in low for s in _VEHICLE_SIGNALS):
        return "vehicle"
    if any(s in low for s in _SPORTS_SIGNALS):
        return "sports"
    if any(s in low for s in _FOOD_SIGNALS):
        return "food"
    if any(s in low for s in _MUSIC_SIGNALS):
        return "music"
    if any(s in low for s in _ANIMAL_SIGNALS):
        return "animal"
    if any(s in low for s in _NATURE_SIGNALS):
        return "nature"
    if any(s in low for s in _HORROR_SIGNALS):
        return "horror"
    if any(s in low for s in _HISTORICAL_SIGNALS):
        return "historical"
    if any(s in low for s in _SCIFI_SIGNALS):
        return "scifi"
    if any(s in low for s in _FANTASY_SIGNALS):
        return "fantasy"
    if any(s in low for s in _CRIME_SIGNALS):
        return "crime"
    if any(s in low for s in _WEATHER_SIGNALS):
        return "weather"
    if any(s in low for s in _URBAN_SIGNALS):
        return "urban"
    if any(s in low for s in _DIALOGUE_SIGNALS):
        return "dialogue_scene"
    person_words = ["man", "woman", "girl", "boy", "person", "people",
                    "he ", "she ", "they ", "character", "figure", "someone"]
    if any(s in low for s in person_words):
        return "sfw_person"
    return "chaos"


# ══════════════════════════════════════════════════════════════════════════════
#  VEHICLE POOLS
# ══════════════════════════════════════════════════════════════════════════════

CARS_RACING = [
    "a 1967 Ford GT40 Le Mans racer, Gulf Oil livery, side exhausts screaming",
    "a Ferrari 488 GT3 in rosso corsa, barely holding the rear through Eau Rouge",
    "a 1969 Dodge Charger Daytona with the massive nose cone and towering rear wing",
    "a modern Formula 1 car, halo glinting under the Abu Dhabi lights",
    "a Porsche 917K in Martini livery, flat-12 howl at 8500 RPM",
    "a McLaren F1 GTR, shark nose low to the tarmac, rain spraying in arcs",
    "a Nissan R390 GT1 on a rain-soaked Mulsanne straight, headlights carving fog",
    "a Toyota Supra MK4 in full JDM street battle trim, brake glow visible",
    "a Bugatti Veyron Super Sport at 250mph on an unrestricted German autobahn",
    "a rally-spec Subaru Impreza WRC98, gravel roostertail twenty feet high",
    "a vintage F1 Lotus 49, mechanic arms waving from the pit wall",
    "a Lamborghini Countach LP5000 QV catching air over a crest at 170mph",
    "a Koenigsegg Jesko Absolut, top speed run at night on an empty highway",
    "a Lancia Stratos HF in Alitalia livery, handbrake turn through a hairpin",
    "a Group B Audi Quattro S1, close enough to taste the fuel",
    "a red Alfa Romeo GTV6 in an underground street race, walls inches away",
    "a 1973 Porsche Carrera RS 2.7, ducktail spoiler, pure analog speed",
    "a Shelby Cobra 427 at the 1965 Le Mans, fenders shaking with each gust",
    "a Renault R5 Turbo in a French mountain ice rally, studs on frozen tarmac",
    "an IndyCar at Indianapolis, banking at 230mph, the g-force visible in the cockpit cam",
    "a Pagani Zonda R on a closed track, titanium exhaust glowing orange at night",
    "a Mazda 787B at Le Mans, the rotary screaming at a pitch no other engine makes",
    "a Mercedes CLR at Le Mans qualifying, front end lifting at speed on the straight",
    "an old-school Top Fuel dragster, 11,000hp, 1000 feet in 3.6 seconds",
    "a 1995 McLaren F1 road car, three seats, naturally aspirated, last of a kind",
]

CARS_STREET = [
    "a midnight-blue 1969 Camaro Z28, exhaust crackling at every light",
    "a Volkswagen Golf GTI MK2, slammed, rolling on rims through a council estate",
    "a Liberty Walk Lamborghini Huracán, wide-body kit millimetres from the kerb",
    "a chrome-wrapped Rolls-Royce Ghost gliding through Beverly Hills at 2am",
    "a beat-up 1985 Chevrolet El Camino, speakers vibrating the windows",
    "a classic cream 1957 Chevrolet Bel Air convertible on a Route 66 sunset",
    "a matte black Dodge Challenger Hellcat, burnout smoke in the rearview",
    "a battered Land Rover Defender, axle-deep in Welsh mountain mud",
    "a lowrider 1964 Chevrolet Impala, hydraulics up at a Sunday show-and-shine",
    "a bright orange 1970 Dodge Challenger R/T, Vanishing Point style",
    "a Lada Niva with a snorkel, fording a Siberian river at dawn",
    "a new Porsche 911 Turbo S in chalk grey, autobahn entry ramp",
    "a custom Ford Transit full build-out camper van, Norwegian coastal road",
    "a 1972 Datsun 240Z in orange, canyon roads in the Santa Monica mountains",
    "a military-spec Toyota Hilux, flat tyre on an African red-dirt track",
    "a 1983 De Tomaso Pantera on a winding Italian mountain road, rear tyres screaming",
    "a Citroën 2CV crossing the Sahara, somehow still running",
    "a Trabant in East Berlin, 1989, the wall just fallen, streets full of people",
    "a stretched Cadillac hearse in New Orleans Second Line traffic",
    "a battered yellow New York taxi, tunnel mouth open ahead at midnight",
]

MOTORCYCLES = [
    "a Kawasaki H2R at 300km/h, the supercharger whine piercing everything",
    "a 1970 Honda CB750 Four, first modern superbike, highway at sunset",
    "a Ducati Panigale V4R, knee scraping at the bottom of a Spanish chicane",
    "a vintage Harley-Davidson Knucklehead, leather saddlebags, Nevada desert",
    "a BMW R90S cafe racer, fairing cracked, going nowhere slowly on a Scottish B-road",
    "a Royal Enfield Bullet 350, monsoon rain, Indian coastal road",
    "a MotoGP Repsol Honda RC213V, rear tyre spinning out of a slow corner",
    "a Vincent Black Shadow, fastest in the world in 1948, still terrifying",
    "a Triumph Bonneville T120, two-up with camping gear, Scottish Highlands",
    "a scrambler-spec Honda CRF450, full send off a sand dune at 100mph",
    "a custom chopper, extended forks, knucklehead engine, Route 66 at sunset",
    "a sidecar BMW Boxer crossing Mongolia, the track invisible in the grass",
]

AIRCRAFT = [
    "an F-22 Raptor in a vertical climb, afterburners punching holes in clouds",
    "a Spitfire Mk IX at 400mph, three German aircraft in the mirror",
    "a Boeing 747-400 rotating off LAX runway 24L into evening marine layer",
    "a Bell 407 helicopter at hover over a Manhattan rooftop helipad",
    "a Red Bull Air Race Extra 300, 10g through a slalom gate at 200mph",
    "a 1969 Concorde prototype, Mach 2.04 over the Atlantic at 60,000 feet",
    "a paramotor at dawn, French vineyards a hundred feet below",
    "a B-2 Spirit stealth bomber, three miles up, completely silent from the ground",
    "an SR-71 Blackbird, skin glowing from aerodynamic heating at Mach 3.2",
    "a glider at 6000 feet, thermal centred, engine-off silence over the Alps",
    "a MiG-29 at an airshow, cobra manoeuvre, speed dropping to zero mid-flight",
    "a seaplane landing on a Norwegian fjord, spray in a perfect arc",
]

WATERCRAFT = [
    "a 1930s Gar Wood V-bottom racing hydroplane, wooden hull hammering at 120mph",
    "an America's Cup AC75 foiling at 50 knots, hull clear of the water",
    "a rigid inflatable boat at full throttle across a 4-metre Atlantic swell",
    "a container ship making 24 knots into the eye of a North Sea storm",
    "a WWII PT boat at speed in the Coral Sea, phosphorescent wake in the dark",
    "a kayak threading a Grade 5 rapid in the Grand Canyon, barely in control",
    "a nuclear submarine ascending from 400 metres, ballast tanks blowing",
    "a solo circumnavigation yacht mid-Southern Ocean, 10-metre seas at night",
    "a vintage Chris-Craft runabout, mahogany gleaming, Lake Tahoe at dawn",
    "a dragon boat racing crew, fifty strokes per minute, the finish 10 metres ahead",
]

VEHICLE_SETTINGS = [
    "at the Nurburgring Nordschleife, trees a green blur at 180mph",
    "on the Circuit de la Sarthe at 3am, Le Mans 24 Hours, Hour 18",
    "on a Norwegian mountain pass, snow walls six feet high either side",
    "in the Rub al Khali desert, Saudi Arabia, nothing on the horizon",
    "on the Transfagarasan road in Romania, switchbacks to 2000 metres",
    "through the Monaco tunnel at the Grand Prix, wall two inches away",
    "on Route 66 near Amarillo, straight to the horizon, heat shimmer rising",
    "on the Paul Ricard circuit, mistral blowing flags horizontal",
    "on the Stelvio Pass with 48 hairpins visible below",
    "through downtown Tokyo at night, Shibuya crossing deserted at 4am",
    "on the Bonneville Salt Flats, white crust stretching fifteen miles",
    "in the Atacama Desert, driest place on Earth, dust devil alongside",
    "in an underground carpark converted to drift track, walls close, echo constant",
    "on the Tail of the Dragon, 318 curves in 11 miles, Tennessee",
    "across the Millau Viaduct in France, 270 metres above the Tarn valley",
    "in a snowstorm on the Grossglockner Hochalpenstrasse, visibility 20 metres",
    "at dawn on an empty German autobahn, no speed limit, doing 300",
    "in the Dakar Rally, Morocco to Senegal, Stage 7, lost and running on feel",
    "at a 1970s Mexican hill climb, spectators literally touching the car",
    "on the 24 Hours of Nurburgring course at 4am, fog rolling in from the Eifel hills",
]

VEHICLE_CAMERA = [
    "helmet-cam POV, visor scratched, g-force pulling the image sideways",
    "low tracking shot from three feet off the ground, wheel arches filling frame",
    "aerial drone locked on, the car threading through the landscape below",
    "cockpit shot through the steering wheel, instruments blurring",
    "chase car camera, twenty metres back, following through the sequence",
    "static wide at the apex of the corner, car approaching, braking, pivoting",
    "slow-motion at 1000fps, debris frozen mid-arc, dust caught mid-explosion",
    "go-pro on the bonnet, road rushing toward the lens at 200mph",
    "infrared night vision, heat blooming from the exhaust and brakes",
    "opposite-lock oversteer, filmed from the barrier looking straight down the slide",
    "pit wall wide shot, car blurring past at full speed between shots",
    "underwater angle through a puddle, splash erupting as the car hits it",
]

# ══════════════════════════════════════════════════════════════════════════════
#  SPORTS POOLS
# ══════════════════════════════════════════════════════════════════════════════

SPORTS_SCENARIOS = [
    "a boxer throwing a perfect right hook in the 12th round, both men bleeding",
    "a snowboarder catching 40 feet of air off a halfpipe lip at the X Games",
    "a surfer bottom-turning inside a 30-foot Nazare wave, white water overhead",
    "a gymnast releasing the high bar mid-Tkatchev, perfectly horizontal",
    "a sprinter hitting top speed at the World Championships, the finish 30 metres away",
    "a MMA fighter sinking a rear-naked choke in the final seconds",
    "a free solo climber at 3000 feet on El Capitan, no rope, nothing below",
    "a ski jumper at the moment of take-off, 140 metres in the air",
    "a football goalkeeper fingertipping a 90mph shot onto the post",
    "a weightlifter clean-and-jerking 250kg, the bar bending visibly",
    "a swimmer touching the wall by 0.001 seconds, the scoreboard flipping",
    "a rally driver catching a snap oversteer at 140mph on gravel",
    "a diver entering the water from 27 metres, nearly zero splash",
    "a BMX rider pulling a 900 in a concrete bowl, crowd going nuclear",
    "a tennis player hitting a 155mph ace on match point at Wimbledon",
    "a wingsuit pilot threading a 30-metre gap in the Dolomites at 250kph",
    "a rodeo rider at 8 seconds on a 1800-pound bull, one hand up",
    "a figure skater landing a quadruple Axel, arms tight to the body",
    "a parkour runner vaulting a 10-storey rooftop gap, nothing below",
    "a pole vaulter clearing 6.21 metres, bar trembling, not falling",
    "a cricket batsman hitting a reverse sweep six over third man at Lord's",
    "a basketball player posterizing a helpless defender at the Finals",
    "a golfer holing a 50-foot eagle putt to win the Masters",
    "a jockey riding a finish at the Grand National, horse length ahead",
    "a kayaker threading the final gate in a whitewater slalom, hundredths apart",
    "a professional skateboarder sticking a trick on the worst possible terrain",
    "a marathon runner at mile 25, everything gone, still moving",
    "a volleyball player pancake-saving a kill shot from three centimetres off the sand",
]

SPORTS_SETTINGS = [
    "a packed Nou Camp at midnight, 90,000 people",
    "an empty Olympic stadium at 6am, one athlete, no witnesses",
    "the Monaco Grand Prix circuit in the rain",
    "a 1960s boxing gym in Brooklyn, bag-work sounds in the empty building",
    "the Alpe d'Huez climb at the Tour de France, thousands lining the road",
    "Wimbledon Centre Court, the shadow line moving across the grass",
    "the half-pipe at Snowmass, Colorado, 2am training session",
    "a Muay Thai ring in a Bangkok backstreet, corrugated iron walls",
    "the final mile of the Boston Marathon, Heartbreak Hill",
    "inside a velodrome, banking at 45 degrees, centrifugal force keeping him up",
    "a rooftop basketball court in New York, summer, no one watching",
    "a Scottish highland games field in the mist, no spectators, just competitors",
]

# ══════════════════════════════════════════════════════════════════════════════
#  ANIMATION / CARTOON POOLS
# ══════════════════════════════════════════════════════════════════════════════

ANIMATION_SCENARIOS = [
    "going on an ill-advised adventure that started with someone saying 'what could go wrong'",
    "discovering a mysterious portal in the back of an ordinary wardrobe",
    "accidentally summoning something that clearly should not have been summoned",
    "racing to stop a villain's plan that accidentally started as a Tuesday",
    "building something that works perfectly until the exact worst moment",
    "getting lost in a new world and befriending the most chaotic local available",
    "entering a competition they are extremely unqualified for",
    "saving the universe by doing something completely ridiculous",
    "following a treasure map drawn by someone who was clearly making it up",
    "defending the town from a threat that is mostly their own fault",
    "infiltrating an enemy base using a plan that somehow only gets stupider",
    "time travelling to fix one small thing and breaking everything else",
    "finding a magic object that grants wishes in the most literal possible way",
    "hosting a party that immediately spirals completely out of control",
    "trying to keep a secret that obviously should not be kept",
    "competing in a grand tournament they signed up for as a joke",
    "exploring the deep ocean or outer space and regretting it immediately",
    "performing a heist on the most secure establishment in the known world",
    "discovering a prophecy about themselves that raises serious questions",
    "chasing something small and unimportant for reasons that have escalated",
    "navigating an enchanted forest that has its own rules and a bad attitude",
    "attending a school for something unusual and immediately breaking every rule",
    "waking up as the villain by accident and trying to course-correct",
    "making a deal with a magical entity who clearly reads contracts better than them",
    "falling into a rival dimension where everything is slightly wrong",
    "accidentally becoming the town hero without doing anything heroic",
    "having to explain something impossible to someone who wants a very simple answer",
    "discovering that the map to the treasure was inside them all along and being furious about it",
]

ANIMATION_WORLDS = [
    "Bikini Bottom — everyone is a sea creature, Krabby Patties are currency",
    "a hand-drawn 1940s studio cartoon world where everything bends and squashes",
    "a Miyazaki-style Japan where spirits live in every object and weather has feelings",
    "Springfield — a yellow-skinned town where nothing changes but everything happens",
    "a Saturday morning superhero universe with terrible villain plans",
    "a rich 2D anime world with dramatic wind and extremely emotive reaction shots",
    "a Pixar-adjacent world where the inanimate has secret feelings",
    "Toontown — pure rubber-hose chaos where physics are a suggestion",
    "a 1980s action cartoon world with laser guns that never actually hit anyone",
    "a Gravity Falls-style Pacific Northwest full of genuine weirdness",
    "a Rick and Morty-style multiverse where every version of reality is worse",
    "a Studio Ghibli forest where the walk to anywhere is the entire point",
    "a video game world in the middle of a speedrun gone wrong",
    "an anime sports academy where every training montage has an orchestral swell",
    "a children's book world that has become corrupted in interesting ways",
    "a fairy tale kingdom where the tropes are self-aware and tired of it",
    "an 80s anime mecha universe where the robots have feelings nobody addresses",
    "a stop-motion universe that is very aware it is stop-motion",
    "a universe where everything is food and nobody asks why",
]

ANIMATION_TONE = [
    "played completely straight as if this is totally normal",
    "maximum cartoon chaos — everything is loud and physical",
    "slow and earnest in a way that makes it oddly moving",
    "extremely dramatic over something completely trivial",
    "absurdist logic — the rules are internal and respected with full commitment",
    "heartfelt and surprisingly emotional despite the setting",
    "action-packed with three separate betrayals",
    "comedy first, but with a genuine emotional gut-punch in the last 30 seconds",
    "philosophical in a way that works precisely because it shouldn't",
    "escalating chaos that ends in the most unhinged possible resolution",
    "deceptively slow, then suddenly everything at once",
    "warm and nostalgic — the comedy is gentle and the stakes are real but small",
]

# ══════════════════════════════════════════════════════════════════════════════
#  FANTASY POOLS
# ══════════════════════════════════════════════════════════════════════════════

FANTASY_SCENARIOS = [
    "a lone knight entering a dragon's cave knowing full well how this usually ends",
    "a court mage accidentally casting a forbidden spell during a state banquet",
    "a dark elf assassin completing a contract and then regretting it immediately",
    "a tavern bard discovering that the legend they've been singing is literally true",
    "a paladin questioning their faith on the longest night of a siege",
    "a hedge witch entering a city where magic is illegal and immediately needing magic",
    "a dwarf engineer detonating something that should not have been detonated",
    "a ranger tracking something through a forest that clearly does not want to be found",
    "a thief breaking into a vault and finding something completely unexpected inside",
    "a healer keeping an entire platoon alive during a battle they cannot possibly win",
    "a necromancer raising an army for reasons that are technically defensible",
    "a shapeshifter maintaining a human cover in a city that knows shapeshifters exist",
    "a fire mage discovering their power has limits at exactly the wrong moment",
    "a sea serpent hunter three days out and realising the ship is too small",
    "a golem given sentience and absolutely no guidance on what to do with it",
    "a vampire lord holding a court of lesser monsters in an ancient manor",
    "a mercenary company doing a job that was not described accurately in the contract",
    "a young dragon learning to fly above clouds that are very, very far down",
    "a time-locked castle where the same moment has been playing for 300 years",
    "a tournament of champions where the prize is something no one actually wants",
    "a cartographer mapping a coastline that changes with the tide in ways that shouldn't be possible",
    "an oracle who has seen the future and is trying not to give too much away",
    "a city of thieves that has elected a thief as mayor and things are not going well",
]

FANTASY_WORLDS = [
    "a crumbling empire on the edge of an ice age, magic dying with the old order",
    "a floating archipelago above the clouds where the sea is sky",
    "a desert city built from the bones of a dead god",
    "a vast underground kingdom that has never seen sunlight in ten generations",
    "a world where magic is industrial and the gods are shareholders",
    "a feudal Japan-adjacent realm where spirits and samurai share the same roads",
    "an endless library that is also a world, every book a different reality",
    "a dying star system where magic is borrowed time and everyone knows it",
    "a single city that is also a country, ruled by sixteen competing guilds",
    "a post-war realm where the winning side is starting to question what they won",
    "a Norse-adjacent cold north where the gods are tired and the monsters are not",
    "an ancient Greek-adjacent world where the myths are current events",
    "a rainforest empire with elemental magic tied to the land and the land is angry",
    "a world where death is reversible and no one can decide if that's good",
    "a city of thieves on the back of a continent-sized creature that is slowly waking",
    "an archipelago where each island is owned by a different dragon and the politics are exhausting",
]

FANTASY_ATMOSPHERE = [
    "torchlit and claustrophobic — every shadow hides something",
    "vast and windswept — the scale is the point",
    "intimate and domestic despite the extraordinary setting",
    "the calm before — everything knows what's coming",
    "aftermath — the battle is over, only the consequences remain",
    "ancient and indifferent — the world predates the characters by millennia",
    "decaying grandeur — everything was once magnificent and is still falling",
    "dense and alive — the world presses in from every direction",
    "celestial and cold — beauty without warmth",
    "rotting and warm — death that is comfortable and familiar",
]

# ══════════════════════════════════════════════════════════════════════════════
#  SCI-FI POOLS
# ══════════════════════════════════════════════════════════════════════════════

SCIFI_SCENARIOS = [
    "a lone engineer doing a spacewalk on the hull of a dying station",
    "first contact — the xenobiologist who trained for this is not ready",
    "a colony ship 400 years from Earth, crew of 12, something has woken up",
    "a time loop detective who has solved this murder 47 times and cannot stop",
    "an AI that has achieved sentience and is deciding whether to mention it",
    "a terraforming crew on Mars, Day 1, the equipment is wrong",
    "a deep-sea research station on Europa — the signal comes from below",
    "a generation ship — the crew born on board have never seen a star up close",
    "last transmission before the relay goes dark — 11 words received",
    "a black market organ dealer on a space station frontier",
    "a soldier discovering mid-mission that they are a clone of someone who died",
    "a city under a dome on a hostile planet — the dome has a crack",
    "an interstellar bounty hunter tracking a target through a baroque space station",
    "a corporate espionage agent downloading secrets, three minutes on the clock",
    "the last human ship in a war that the machines stopped bothering to fight",
    "an android waking up in a scrapyard with memories that are not theirs",
    "a wormhole transit that deposited the ship somewhere new and the nav charts are wrong",
    "a resistance fighter in a city where every surface is a surveillance screen",
    "first day of alien occupation — the invaders are not what the media said",
    "a generation of humans born in space, returning to Earth for the first time",
    "a xenoarchaeologist on a planet that was Earth 50,000 years ago",
    "a quantum mechanic maintaining a drive that should not exist according to physics",
]

SCIFI_SETTINGS = [
    "a Blade Runner-esque megacity, twelve-storey neon signs, rain permanent",
    "a derelict space station in a debris field, emergency lighting only",
    "the surface of Titan — orange haze, methane lakes, absolute silence",
    "a corporate arcology — 200 floors, entire ecosystem inside glass and steel",
    "a generation ship's industrial level — pipes, reactors, workers who have never been to the bridge",
    "an alien planet with two suns and completely unfamiliar geology",
    "a digital mindspace — the simulation is indistinguishable but something is off",
    "low Earth orbit — the planet below, catastrophically visible",
    "a jungle planet — ancient ruins from a civilisation that left no other trace",
    "an orbital elevator — the tether stretching 36,000km into the dark",
    "a warship bridge at battle stations — every screen showing something worse",
    "a cryogenic bay where the sleepers are waking up 200 years early",
    "a xenolinguistics lab — the alien signal is decipherable but the meaning is ambiguous",
    "a quantum computing facility with an AI running at 10^24 operations per second",
    "a bio-dome on Mars, red rock visible through every window, Earth a bright star at night",
]

SCIFI_TONE = [
    "cold and technical — the language is precise and the stakes are enormous",
    "human and warm despite the inhuman setting",
    "dread building slowly from the first frame",
    "action-first, consequences later",
    "philosophical — the technology is window-dressing for the real question",
    "noir — everyone is lying, the truth is worse than expected",
    "satirical — the future is corporate and petty and familiar",
    "survival horror — the technology cannot help with this",
    "bittersweet — the ending is ambiguous on purpose",
    "awe and terror in equal measure — the scale is the point",
]

# ══════════════════════════════════════════════════════════════════════════════
#  ANIMAL POOLS
# ══════════════════════════════════════════════════════════════════════════════

ANIMAL_SCENARIOS = [
    "a wolf pack circling prey in the first snowfall of winter",
    "a humpback whale breaching twice its body length out of the ocean",
    "a peregrine falcon in a 250mph stoop, target locked from 300 metres",
    "a pride of lions at a waterhole at dusk, flies settling on their manes",
    "a great white shark at the moment of breach, clear of the water, seal in jaws",
    "a polar bear swimming 50 miles of open Arctic water to reach ice",
    "a silverback gorilla at rest, watching something only he can see",
    "a cheetah at full stride — 112km/h, two strides per second",
    "a mantis shrimp striking at 23 metres per second, cavitation bubble forming",
    "an elephant matriarch leading 40 across a dry savannah toward water she remembers",
    "a barn owl hunting by sound in complete darkness, wings silent",
    "a komodo dragon ambushing a deer — two hours of patient waiting",
    "a blue whale — 30 metres of life, longer than a tennis court, breathing once",
    "a cuttlefish changing colour and pattern in 300 milliseconds to match coral",
    "a murmuration of 100,000 starlings over the Ebro delta at dusk",
    "an orca coordinating a wave-wash hunt with three other family members",
    "a jumping spider examining a camera lens with the same curiosity it is being examined with",
    "a honey badger ignoring three lion cubs with complete disregard for the situation",
    "a migratory bird covering 12,000km on its first journey, guided by magnetism",
    "a pufferfish inflating to six times normal size when the diver gets too close",
    "a golden eagle taking a red fox on a Welsh hillside, impact visible from 500 metres",
    "a mother bear breaking ice to reach salmon, cubs watching her technique",
    "a narwhal pod moving under ice in formation, tusks pointed forward",
    "a secretary bird stomping a puff adder to death with methodical precision",
]

ANIMAL_SETTINGS = [
    "Serengeti at the edge of a storm, lightning on the horizon, grass moving",
    "a boreal forest in November, two feet of snow, silence broken by one crack",
    "the Mariana Trench, 11km down, bioluminescent and crushing",
    "a coral reef in the Red Sea at maximum visibility, warm and saturated",
    "the Arctic at the summer solstice — 24 hours of low golden light",
    "a Costa Rican rainforest at dawn, the first light finding the canopy",
    "a Galapagos beach — marine iguanas, sea lions, nothing afraid of anything",
    "a Scottish Highland loch in February, everything grey and clear and cold",
    "an African waterhole at 2am — infrared, eyes everywhere in the dark",
    "the Bering Sea crab season — 50-knot winds, 10-metre seas, deck work",
    "a Kenyan savannah just before the wildebeest migration hits the Mara river",
    "a Norwegian fjord in winter, orca dorsal fins cutting the surface",
]

# ══════════════════════════════════════════════════════════════════════════════
#  FOOD / COOKING POOLS
# ══════════════════════════════════════════════════════════════════════════════

FOOD_SCENARIOS = [
    "a ramen master pulling noodles by hand at 3am, broth at 18-hour simmer",
    "a pastry chef laminating 729 layers of croissant dough in a 4C kitchen",
    "a sushi itamae pressing a piece of toro nigiri with exactly the correct pressure",
    "a taco cart operator feeding 200 people from a single griddle during a festival",
    "a Neapolitan pizzaiolo shaping a Margherita in 14 seconds from muscle memory",
    "a molecular gastronomist spherifying olive oil into caviar at a two-star kitchen",
    "a home baker pulling their first successful sourdough from a Dutch oven",
    "a competition BBQ team at 4am tending a 14-hour brisket in the cold",
    "a Georgian feast being assembled — 40 dishes on a table for 20 people",
    "a fisherman breaking down a 60kg tuna on the Tsukiji market dock",
    "a French grandmother making a cassoulet that takes two days and no shortcuts",
    "a street food vendor in Bangkok who has made one dish perfectly for 30 years",
    "a Michelin kitchen brigade during a Saturday night service, 200 covers",
    "a cheese affineur turning wheels of 24-month Comte in a limestone cave",
    "a coffee roaster dialling in a single-origin Ethiopian at the moment the first crack breaks",
    "a dim sum kitchen at 6am, the folding is muscle memory and it shows",
    "a chocolatier tempering by hand at body temperature, touching the marble slab",
    "a bread baker scoring a batard at 5am, oven preheated since 3",
    "a Cantonese wok chef hitting 400C with a flash of oil, flame shooting two feet",
]

FOOD_SETTINGS = [
    "a French farmhouse kitchen, copper pots on hooks, window open to the garden",
    "a Tokyo ramen bar, eight seats, steam and pork fat in the air",
    "a competition kitchen — clock visible, judges watching, nothing going to plan",
    "a market in Marrakech, spices in cones, light through the souk lattice",
    "a family home in Naples, Sunday lunch, everyone slightly in the way",
    "a cutting-edge restaurant kitchen, all white and stainless, complete silence",
    "a Vietnamese banh mi shop opening at 5am, bread still in the oven",
    "a basement speakeasy kitchen feeding an illegal supper club",
    "a rural smokehouse on the west coast of Ireland, smell of salt and wood",
    "a spice market in Chennai at 7am, the colour overwhelming",
    "a mountain hut in the Swiss Alps, one wood stove, four dishes that have been made here for 80 years",
]

# ══════════════════════════════════════════════════════════════════════════════
#  HORROR POOLS
# ══════════════════════════════════════════════════════════════════════════════

HORROR_SCENARIOS = [
    "a paranormal investigator opening a door they should leave closed",
    "a small town where everyone is friendly and no one mentions what lives in the lake",
    "a haunted house that moves its rooms between visits",
    "a lighthouse keeper in week three — the log is getting strange",
    "a plague doctor in 1348 who has started to suspect this is not a disease",
    "a sleep study participant who is definitely the only one still awake",
    "an archaeologist translating something that should not be translatable",
    "a family moving into a house that still has the last family's things in it",
    "a child describing their imaginary friend to a therapist who is increasingly concerned",
    "an Antarctic expedition where something came back with the drilling team",
    "a submersible at depth, outside pressure nominal, inside pressure wrong",
    "a night security guard in a museum where the cameras go dark at the same time each night",
    "a town that loses one day of memory every year and no one knows why",
    "a delivery driver making the last stop on a route that wasn't on the manifest",
    "a cult compound — the members are happy and the happiness is the horror",
    "a hospital where the night shift nurses don't cast shadows",
    "a forest where the trees have grown over a road that no longer exists on maps",
    "a 1970s vacation film, pastoral and warm, until something in the background moves",
    "a radio host receiving a transmission from a station that closed in 1987",
    "a neighbourhood where no one ages and the calendar stopped at the same Tuesday twenty years ago",
]

HORROR_SETTINGS = [
    "a deconsecrated church in rural England, three hours from the nearest town",
    "a 1920s psychiatric hospital, everything still exactly where it was abandoned",
    "a long service corridor with fluorescent lights at the far end flickering",
    "a fog-bound New England fishing village, off-season, no one on the dock",
    "a Japanese ryokan in a mountain valley, the other guests checked out overnight",
    "an airport at 3am, one gate still showing a departure, no one at the desk",
    "a children's hospital ward, the TV on in the corner playing nothing scheduled",
    "a roadside motel with 12 rooms and always one car more than there should be",
    "an abandoned theme park, rides still powered, no one able to explain why",
    "a rural American diner with six booths and a door to the back that staff don't open",
    "a suburban house where the lights in the basement have been on for six months",
    "an island that appears on old maps but not new ones, and there are fresh footprints",
]

HORROR_TONE = [
    "slow dread — nothing happens for a long time and that is the horror",
    "folk horror — the community is the threat and the community is kind",
    "cosmic — the thing is not evil, it simply does not register your existence",
    "domestic — the horror is inside the ordinary and that is why it works",
    "body horror — the transformation is internal and unstoppable",
    "psychological — the line between real and imagined is the story",
    "survival — immediate, physical, no time for subtext",
    "grief — the horror is a vehicle for something that actually happened",
    "atmospheric — the sound design is doing 80% of the work",
    "satirical horror — the monster is obvious metaphor and knows it",
]

# ══════════════════════════════════════════════════════════════════════════════
#  HISTORICAL POOLS
# ══════════════════════════════════════════════════════════════════════════════

HISTORICAL_SCENARIOS = [
    "a Roman centurion holding a line at the Battle of Cannae as everything collapses",
    "a Viking longship making landfall on an unknown coast — the crew doesn't know it's America",
    "a samurai in the moment before a duel, opponent 20 metres away, autumn leaves",
    "a WWII Spitfire pilot in a dogfight over the Channel at 22,000 feet",
    "a gold rush prospector finding a 30-pound nugget and knowing what that means",
    "a medieval monk illuminating a manuscript by candlelight as plague reaches the city gates",
    "an Ottoman siege engineer calculating the angle for the largest cannon ever built",
    "a geisha performing tea ceremony in Kyoto 1850, three years before Perry arrives",
    "a gladiator entering the Colosseum for the 40th time, crowd of 50,000, no longer afraid",
    "a first world war soldier going over the top at 0600 with a whistle blast",
    "an Egyptian scribe recording a pharaoh's death, knowing history is being changed",
    "a Mongol horseman at the Mohi bridge Hungary 1241, the most feared cavalry ever assembled",
    "a court astronomer in Renaissance Florence showing a painting to Lorenzo de' Medici",
    "a pirate captain in the Caribbean in 1718, Nassau on the horizon, Blackbeard at the wheel",
    "a coal miner coming up from the pit on the last day before the machine replaces them",
    "a Native American elder at the moment first seeing a European ship on the horizon",
    "a Spartan at Thermopylae on day two, Leonidas still standing",
    "a British redcoat in the American colonies on the night of April 18 1775",
    "a Soviet cosmonaut on Sputnik reentry, heat shield unknown status, six minutes to know",
    "a Berlin Wall guard in November 1989 watching the crowds arrive and making a decision",
]

HISTORICAL_SETTINGS = [
    "the Forum Romanum at its peak — white marble blinding in July sun",
    "a Viking longhouse in Norway 900 AD, winter pressing the walls",
    "a WWI trench at dawn, mud and rain and the silence before the whistle",
    "an Egyptian temple complex at construction — 20,000 workers, dust everywhere",
    "Edo-period Kyoto, paper lanterns at dusk, the wooden city",
    "a medieval market town 1347, a fortnight before the Black Death arrives",
    "a Wild West frontier town 1880, one marshal, forty men he can't trust",
    "the Somme battlefield 1916, between offensives — exhausted and completely still",
    "a 1920s jazz speakeasy, Prohibition America, someone always watching the door",
    "Constantinople 1453, the last night before the walls fall",
    "the Apollo 11 mission control Houston, 3am local time, moon landing four hours away",
]

# ══════════════════════════════════════════════════════════════════════════════
#  NATURE / LANDSCAPE POOLS
# ══════════════════════════════════════════════════════════════════════════════

NATURE_SCENARIOS = [
    "a solo climber on the North Face of the Eiger, day two, weather changing",
    "a kayaker entering a sea cave on the Faroe Islands as the tide comes in",
    "a storm chaser 200 metres from an EF5 tornado, tripod shaking",
    "a photographer waiting for the green flash at sunset over the Pacific",
    "a geologist in a canyon where every layer is a different millennium",
    "a diver at the thermocline — warm above, cold below, visibility to 40 metres each way",
    "a forager in an old-growth forest finding something that shouldn't exist this far south",
    "a shepherd in the Transylvanian Alps during the great annual migration of flocks",
    "an aurora watcher on the Lofoten Islands as a G4 storm peaks",
    "a glaciologist listening to a glacier calve — the sound five seconds before the sight",
    "a botanist discovering an undescribed species in a cloud forest in Ecuador",
    "a desert photographer in the Namib at dawn, the dunes orange and shadowless",
    "a solo kayaker rounding Cape Horn in a 6-metre swell, the southernmost point",
    "a bioluminescent bay at 2am — every movement in the water leaving blue fire",
    "a volcanologist 80 metres from an active lava flow, heat shimmer distorting everything",
    "a cave diver at the limit of visibility in a sump 200 metres underground",
]

NATURE_SETTINGS = [
    "Trolltunga Norway — a rock platform hanging 700 metres above a glacial lake",
    "the Okavango Delta at flood — 15,000 square kilometres of water in the Kalahari",
    "Zhangjiajie China — sandstone columns rising 300 metres from a sea of mist",
    "the Saharan Grand Erg Occidental — dunes 180 metres high, not a track anywhere",
    "the Dolomites in a thunderstorm, every peak lit from below by lightning",
    "the Amazon at the point where the black water and white water rivers don't mix",
    "Socotra Island — dragon blood trees in full canopy, an alien landscape on Earth",
    "the midnight sun in Svalbard — gold light at 2am, polar bear tracks in the snow",
    "a supercell thunderstorm over the Great Plains, 20 kilometres wide, rotating",
    "the Zhangye Danxia landform — red and orange and yellow banded hills like a painting",
    "the Waitomo glowworm caves, New Zealand — a ceiling of living blue stars",
]

# ══════════════════════════════════════════════════════════════════════════════
#  MUSIC POOLS
# ══════════════════════════════════════════════════════════════════════════════

MUSIC_SCENARIOS = [
    "a guitarist playing the same phrase for the 900th time until something unlocks",
    "a conductor bringing in the full orchestra at the exact right moment",
    "a rapper freestyling to a crowd that has started to lean forward",
    "a jazz pianist finding a run at 2am that they've been chasing for six months",
    "a DJ reading the room and changing everything on the last 8 bars",
    "a classical violinist mid-concerto when the bow hair starts to fail",
    "a blues singer at a dive bar, 11pm Tuesday, the six people left listening",
    "a producer at 3am when the sample finally fits the way they heard it in their head",
    "a street busker on the London Underground playing Bach on a battered violin",
    "a festival main stage at the moment the chorus of the last song drops",
    "a choir of 200 voices locking into a chord and holding it for nine bars",
    "a session bassist recording the same 4-bar part for the 60th take",
    "a classical composer hearing their symphony performed for the first time",
    "a record store owner hearing something through the wall from the back that has to be played",
    "a drummer in a marching band, rain soaking through the snare skin, one mile left",
    "a cellist practicing a Shostakovich concerto alone in a rehearsal room at midnight",
    "a Flamenco dancer waiting in the wings, the guitar already starting, ten seconds",
]

MUSIC_SETTINGS = [
    "Abbey Road Studio Two — the same floor the Beatles used",
    "a jazz club in New Orleans at 1am, bourbon and brass",
    "Glastonbury Pyramid Stage at night, 100,000 people, light rain",
    "a subway platform in New York, the echo carrying the saxophone six cars",
    "a Baroque concert hall in Vienna, crystal chandeliers, no microphones",
    "a hip-hop cipher in a Bronx carpark, freestyle, no filter",
    "a church in Harlem, Sunday morning, the organ and the choir",
    "a recording studio control room, tape rolling, everything riding on this session",
    "a basement venue in Berlin at 6am, 40 people left, the DJ has found something",
    "Carnegie Hall at full house, pre-concert silence that has weight to it",
    "a folk session in a pub in County Clare, the door open to the rain",
]

# ══════════════════════════════════════════════════════════════════════════════
#  SFW PERSON POOLS
# ══════════════════════════════════════════════════════════════════════════════

SFW_PERSON_SCENARIOS = [
    "exploring an abandoned train station in Eastern Europe, torch in hand",
    "crossing a rope bridge over a 300-metre gorge in Bhutan",
    "haggling in a souk in Marrakech at the start of a two-month overland trip",
    "surfing a wave they had no business attempting and barely making it",
    "sitting in a Tokyo ramen bar at midnight not speaking the language but pointing correctly",
    "arriving at an airport in the wrong city because they booked it wrong",
    "running the final kilometre of their first ultramarathon on fumes",
    "photographing a lightning storm from a hotel balcony in South Africa",
    "watching a murmuration of starlings from a bridge over the Thames at dusk",
    "cooking a multi-course dinner for strangers they met six hours ago",
    "at the base of a 3000-year-old tree realising scale for the first time",
    "in a packed train in Mumbai at rush hour, completely out of place and fine with it",
    "reading the last page of a book on a train that's about to pull into the final station",
    "getting the film photograph developed and finding one frame they don't remember taking",
    "playing chess in a Buenos Aires park against someone who hasn't lost in six years",
    "watching their home country's coastline disappear as the ship rounds the headland",
    "landing a plane for the first time solo, instructor on the ground, hands empty",
    "speaking a language badly enough that everyone is charmed by the attempt",
    "getting completely lost in a foreign city and having the best day because of it",
    "meeting someone interesting at the exact moment everything else goes wrong",
    "sitting on a rooftop in Havana playing dominoes with three strangers at midnight",
    "watching a sunrise from a temple in Angkor Wat with thirty other people who all kept quiet",
    "finishing something they started three years ago and not knowing what to feel",
    "eating the best meal of their life in a place they can't find again on the map",
]

SFW_DESTINATIONS = [
    "rural Japan in the week of sakura",
    "Lagos during Afrobeats festival season",
    "the streets of Havana in 1958 — forever",
    "a Scottish island with 200 people and no mobile signal",
    "northern Norway in full aurora season",
    "the Faroe Islands in a horizontal rain squall",
    "Buenos Aires on a Sunday when the whole city dances",
    "Tbilisi Georgia, first night, wine and bread and strangers",
    "the Atacama Desert during the flower bloom that happens every decade",
    "rural Morocco in the week of Eid al-Adha",
    "Varanasi at dawn, Ganges fog, a city 3000 years old still running",
    "Oaxaca Mexico, Day of the Dead week, marigolds everywhere",
    "a remote monastery in Bhutan three days' walk from the nearest road",
    "New Orleans in the week before Mardi Gras, before the tourists outnumber the residents",
    "Kolkata at the Durga Puja — 10 days, 150,000 street installations, six nights that never sleep",
    "a roadside guesthouse in Laos with no reason to leave for three days",
    "Lisbon at 3am, the fado still audible through a tiled alley, nobody in a hurry",
    "a cargo ship crossing the Pacific, 22 days, 12 passengers, complete disconnection",
]

SFW_MOODS = [
    "completely absorbed in the moment, no phone in sight",
    "the slight disorientation of being genuinely somewhere new",
    "exhausted in the specific way that is also deeply satisfied",
    "quietly triumphant about something no one else would understand",
    "the alertness of being completely outside comfort zone and enjoying it",
    "the specific calm of someone who has made a decision and stopped doubting it",
    "the concentration of someone doing something they've done 10,000 times",
    "the mild panic of someone doing something for the first time in public",
    "the warmth of a conversation that needed no shared language",
    "the specific joy of finding something you weren't looking for",
    "the stillness of someone who has run out of things to worry about for a moment",
    "the barely-contained excitement of someone who knows something extraordinary is about to happen",
]

# ══════════════════════════════════════════════════════════════════════════════
#  DIALOGUE POOLS — manufacture spoken words for scenes
# ══════════════════════════════════════════════════════════════════════════════

DIALOGUE_OPENERS = [
    "\"You weren't supposed to be here.\"",
    "\"I've been waiting three years to say this.\"",
    "\"Don't. Don't say anything. Just listen.\"",
    "\"The last time I saw you, you told me everything would be fine.\"",
    "\"I'm not angry. That's the thing. I'm just done.\"",
    "\"You knew. The whole time, you knew.\"",
    "\"Tell me one true thing. Just one.\"",
    "\"I have about thirty seconds before this becomes irreversible.\"",
    "\"If I asked you to walk away right now, would you?\"",
    "\"Say that again. Slower. Look me in the eye and say it again.\"",
    "\"We don't have to do this.\" — \"Yes we do.\"",
    "\"This isn't what I wanted.\" — \"Then what did you want?\"",
    "\"I kept your number. I never called it. I kept it.\"",
    "\"What did you think was going to happen?\"",
    "\"I'm not asking for permission. I'm telling you what's happening.\"",
    "\"There's a version of this where we both walk out.\"",
    "\"You were supposed to stop me.\"",
    "\"Every single decision I made was because of you.\"",
    "\"I know what I look like right now. I don't care.\"",
    "\"The world ends in four minutes. Say something real.\"",
]

DIALOGUE_SUBTEXT = [
    "they're not saying what they mean and both characters know it",
    "one character is lying, the other is deciding whether to call it",
    "the conversation is about one thing but entirely about something else",
    "every word is chosen carefully because the wrong one ends everything",
    "they've had this conversation before and it ended badly",
    "one of them already knows how this ends",
    "there's something neither of them will say out loud",
    "the silence between lines is carrying more weight than the words",
    "they are deeply in love and have never said it",
    "one character is saying goodbye and the other doesn't know it yet",
    "they are the only two people in the world right now and they both feel it",
    "the disagreement is genuine — neither person is wrong",
]

DIALOGUE_DELIVERY = [
    "quiet and very controlled — the restraint is the performance",
    "one voice breaking on the last word",
    "fast and overlapping — neither waiting for the other to finish",
    "one monologue, the other person completely still",
    "the pause before the answer is doing all the work",
    "delivered through tears that are refused the whole time",
    "dark humour cutting through genuine grief",
    "flat and exhausted — past the point of emotion into something quieter",
    "the words are calm but the hands give it all away",
    "the last sentence barely audible, barely held together",
]

DIALOGUE_CONTEXTS = [
    "a two-person scene lit by a single lamp in an otherwise dark room",
    "through a door — one person inside, one person in the hall, neither moving",
    "a car at a red light that neither person wants to go green",
    "a phone call where one person has already decided something",
    "across a kitchen table at 3am, two cups of coffee going cold",
    "a hospital corridor where one person has been standing for six hours",
    "a rooftop — the city below, the conversation above everything",
    "a goodbye at a train platform with 90 seconds on the clock",
    "a restaurant where one person has rehearsed this and it's going wrong",
    "standing at the edge of something — a cliff, a bridge, a decision",
    "the back seat of a taxi neither person booked",
    "a voicemail that is too long and hasn't been deleted",
]

# ══════════════════════════════════════════════════════════════════════════════
#  CRIME / NOIR POOLS
# ══════════════════════════════════════════════════════════════════════════════

CRIME_SCENARIOS = [
    "a heist team discovering mid-job that the safe has someone inside it",
    "a hitman sitting across from their target in a diner, neither moving",
    "a detective interviewing the only witness who also did it",
    "a carjacker realising the car they've stolen is carrying something they shouldn't know about",
    "a corrupt cop being turned by a fed who has everything on tape",
    "a pickpocket on the Tube who accidentally took the wrong wallet",
    "a getaway driver who has been parked for 40 minutes and the crew is not coming",
    "a forensic accountant finding the one number that unravels everything",
    "an undercover officer three years in, genuinely unsure which side they're on",
    "a blackmailer opening the envelope and finding something they didn't expect",
    "a fence examining merchandise they realise they recognise",
    "a mob enforcer receiving an order they won't carry out",
    "a con artist running the same mark for the third time without meaning to",
    "a snitch in a holding cell working out how long they have",
    "a safecracker thirty seconds from the alarm — wrong combination, last try",
    "a gang accountant discovering their boss has been skimming from himself",
    "a PI following someone who has just started following them back",
    "a retired criminal recognising a face on the evening news",
]

CRIME_SETTINGS = [
    "a rain-soaked industrial dock at 2am, single floodlight, two cars",
    "a 1950s LA detective's office — fan, bourbon, and a knock at the door",
    "a high-rise casino's private counting room, three hours before the skim",
    "an interview room — grey walls, one mirror, one light, two chairs",
    "a pawn shop on a dead street that has never had a sign in the window",
    "a warehouse somewhere in the port where the postcode changes nothing",
    "a private members' club where the membership is never discussed",
    "a phone booth in the rain — last call before something irreversible",
    "a penthouse with a view and a body that needs to not be found by morning",
    "a diner at the edge of the city where nobody eats, they just sit",
    "the backseat of a blacked-out SUV moving through unknown streets",
    "a decommissioned police station being used for something the deed doesn't cover",
]

CRIME_TONE = [
    "classic noir — everything is moral compromise and cigarette smoke",
    "procedural and cold — the facts are all that matter",
    "heist — the plan was perfect until it wasn't",
    "psychological — the crime is secondary to what it cost",
    "urban thriller — the city is as much a threat as the antagonist",
    "darkly comic — nobody meant for it to go this far",
    "revenge — the motive is clean even if the execution isn't",
    "cat and mouse — both players are equally matched and know it",
]

# ══════════════════════════════════════════════════════════════════════════════
#  EXTREME WEATHER POOLS
# ══════════════════════════════════════════════════════════════════════════════

WEATHER_SCENARIOS = [
    "a storm chaser abandoning the truck as an EF5 drops 400 metres ahead",
    "a surfer paddling into a 60-foot wave at Nazaré while the cliff watches",
    "a mountaineer in a whiteout on the Hillary Step — zero visibility, 8800 metres",
    "a fishing vessel captain deciding whether to run or ride out a Force 11",
    "a wildfire crew cutting a firebreak as the front moves faster than predicted",
    "a flood rescue team in an inflatable working a submerged street at night",
    "a volcano observatory scientist evacuating when the seismograph hits the stop",
    "a skydiver exiting at 15,000 feet into a supercell that wasn't on the forecast",
    "a lone cyclist on an open road when the funnel cloud appears on the horizon",
    "a camper waking in a tent to find themselves surrounded by flash flood water",
    "a lightning photographer on a hilltop as the storm decides to centre overhead",
    "a snow-roller in a Norwegian fjord, avalanche track crossing the only road",
    "a coast guard crew launching into Force 9 swells for a MAYDAY 40 miles out",
    "a meteorologist on live TV as the studio loses power mid-broadcast",
]

WEATHER_SETTINGS = [
    "the Great Plains during tornado season — sky green, rotation visible",
    "the Atlantic seaboard as a Category 4 makes landfall at high tide",
    "the Himalayas above base camp as the jet stream drops without warning",
    "the Australian outback during a mega-fire — the horizon is a wall of orange",
    "the Gulf of Mexico in a Category 5, 300 miles from landfall",
    "a Norwegian fjord town where the avalanche is already visible on the mountain",
    "the Sahara during a haboob — a 3000-foot wall of sand moving at 40mph",
    "the Philippines during typhoon season — the eye passing is the dangerous quiet",
    "a European city during a once-in-200-years flood, water at first floor height",
    "Iceland during a volcanic eruption — ash cloud expanding above the glacier",
]

WEATHER_ATMOSPHERE = [
    "the specific light that only exists when a tornado is forming — green and still",
    "everything moving except the eye of the storm, which is a different universe",
    "the pressure drop before the lightning — ears popping, hair rising",
    "sound first — the freight-train roar before anything is visible",
    "the false calm at the centre, knowing the back wall is coming",
    "the aftermath: total silence, the air wrong, everything rearranged",
    "the decision moment: run or document — 10 seconds to choose",
]

# ══════════════════════════════════════════════════════════════════════════════
#  URBAN / STREET LIFE POOLS
# ══════════════════════════════════════════════════════════════════════════════

URBAN_SCENARIOS = [
    "a graffiti artist working a 40-foot freight train between 2am and 4am",
    "a late-night kebab van owner who has heard everything and remembered it all",
    "a bicycle courier threading 40mph through stationary traffic with one second's margin",
    "a street photographer catching the fraction of a second that tells a life story",
    "a nightclub bouncer deciding who gets through on a Saturday at midnight",
    "a last-night-of-the-venue DJ playing to 200 people in a building being demolished Monday",
    "a food delivery rider at 3am who has memorised every pothole on the route",
    "a window cleaner at height on a glass tower with an unrestricted view of everything",
    "a pawnbroker appraising an heirloom at 9am from someone who drove through the night",
    "a taxi driver on the last fare of a 14-hour shift with something on their mind",
    "a corner shop owner unlocking at 5am — the same six customers in the same order",
    "a building security guard at 4am watching 87 cameras and one of them just moved",
    "a street artist turning a derelict building into something that makes people stop",
    "a homeless man on the same bench for eleven years who is also the most interesting person there",
    "a club promoter trying to fill 500 seats by midnight — it's 11:43pm",
    "a late-shift barista in a 24-hour coffee shop where the clientele is entirely interesting",
]

URBAN_SETTINGS = [
    "east London on a Saturday at 2am — six languages in a single block",
    "a New York subway platform at rush hour where everyone is performing something",
    "a Hong Kong night market at 10pm — a hundred stalls and everything edible",
    "the Shibuya crossing in Tokyo at peak hour — 3000 people per cycle",
    "an Istanbul bazaar at dawn being set up — the city before the city",
    "a São Paulo favela at Carnaval — the geography makes the sound different",
    "a Chicago L-train at 6am, everyone going somewhere that matters to them",
    "a Soho alley in the 1980s — clubs, dealers, record stores, six floors of decisions",
    "a Detroit neighbourhood with one extraordinary bar still open since 1962",
    "a Paris street at 3am — silent, orange-lit, still somehow alive",
    "a Nairobi matatu at rush hour — music up, no air, 22 people in 14 seats",
    "a Mumbai street food strip at midnight — 40 vendors, one block, no gaps",
]

URBAN_ENERGY = [
    "hyperkinetic — the city is making a sound that is all its sounds at once",
    "liminal — the time of day when the city belongs to different people",
    "documentary — observe without interference, everything is already interesting",
    "neon-lit and sleepless — this is the city that doesn't acknowledge sunrise",
    "human-scale — the city zoomed to one face, one decision, one moment",
    "the specific loneliness of being surrounded by 8 million people",
    "gritty and alive — the imperfection is the point",
]

# ══════════════════════════════════════════════════════════════════════════════
#  CHAOS MODE — full universe list
# ══════════════════════════════════════════════════════════════════════════════

CHAOS_UNIVERSES = [
    "vehicle", "sports", "animation", "fantasy", "scifi",
    "animal", "food", "horror", "nature", "music",
    "historical", "sfw_person", "crime", "weather", "urban",
    "dialogue_scene",
]

# ══════════════════════════════════════════════════════════════════════════════
#  UNIVERSAL MODIFIERS
# ══════════════════════════════════════════════════════════════════════════════

UNIVERSAL_VISUAL_STYLE = [
    None, None, None,
    "cinematic 35mm film, visible grain, slightly underexposed",
    "golden hour backlight, lens flare, long shadows",
    "neon-soaked, deep shadows, cyan and magenta palette",
    "crisp documentary — handheld, natural light, no grade",
    "moody Rembrandt lighting — one source, deep shadow",
    "desaturated and gritty — the colour is wrong on purpose",
    "VHS aesthetic — tracking lines, color bleed, era correct",
    "high contrast monochrome — print grain, deep blacks",
    "teal and orange Hollywood grade",
    "soft romantic diffusion — dreamy and warm",
    "infrared — vegetation white, sky black, skin glowing",
    "anamorphic — oval bokeh, horizontal lens flare, scope ratio",
    "1970s Kodachrome — saturated greens and warm reds",
    "hyperrealistic — every pore, every fibre, every surface",
    "impressionistic — the edges are soft and the colour is the story",
    "flat lay top-down — everything composed on a surface",
    "Wes Anderson symmetry — dead centre, warm palette, every prop deliberate",
    "Roger Deakins naturalism — light from where light actually comes from",
    "Super 8mm home movie — warm, washed, edges slightly vignetted",
    "daguerreotype grain — the image fighting to stay fixed",
    "IMAX ratio — the frame is enormous and the subject earns it",
    "bleach bypass — silver retained, contrast brutal, colour almost gone",
    "day-for-night — shot in daylight, graded cold and blue",
    "split diopter — two planes of focus, both sharp, both present",
    "photojournalism — no artifice, no lighting, just what was there",
    "baroque chiaroscuro — darkness is 70% of the frame",
    "pastel-washed — everything soft, no hard edges, faded summer",
    "Fuji Velvia oversaturation — greens electric, skies impossible",
]

UNIVERSAL_CAMERA = [
    None, None,
    "extreme close-up — the detail is the subject",
    "wide establishing shot — the scale is the story",
    "low angle looking up — power in the geometry",
    "aerial — looking down, everything small and patterned",
    "POV — first person, the viewer is there",
    "over-the-shoulder — following, slightly behind",
    "tracking shot — moving parallel to the subject",
    "slow zoom in — the world contracting to one thing",
    "rack focus — one thing sharp, another becoming sharp",
    "handheld — slight movement, everything real",
    "Dutch angle — the world is slightly wrong",
    "symmetrical — perfect balance, Kubrick-composed",
    "macro — the world at 10x, familiar things alien",
    "steadicam glide — smooth and present, never still",
    "long lens compression — everything flattened, distance collapsed",
    "whip pan — subject found mid-motion, the world a blur behind it",
    "oner — the camera never cuts, the scene breathes in real time",
    "two-shot — both faces in frame, both present simultaneously",
    "insert cut — the detail that tells you everything",
    "crash zoom — the 70s punch, aggressive and committed",
    "tilt-shift miniature — life-size made to look like a model",
    "periscope angle — ground-level, everything towering",
    "bird's eye locked-off — the world arranges itself below",
    "follow-focus racking — the scene reorganising its priorities",
]

UNIVERSAL_TIME = [
    None,
    "golden hour — the last 20 minutes of warm light before it's gone",
    "blue hour — the 10 minutes after sunset when everything is cool and even",
    "3am — the specific quality of a world that has stopped except for this",
    "midday harsh — no shadows, nowhere to hide, all surface",
    "pre-dawn — the sky is lighter than the land but barely",
    "overcast bright — the whole sky is a softbox",
    "during a thunderstorm — light wrong, pressure changed",
    "just after rain — wet surfaces doubling every light source",
    "full moon — enough light to see by, colour stripped out",
    "first snow — before footprints, the world simplified",
    "the moment before the sun clears the horizon — one minute of pure rose gold",
    "magic hour second pass — the sky goes pink then mauve then nothing for fifteen minutes",
    "noon in winter — the sun never high, shadows always long",
    "industrial night — sodium orange everywhere, no natural light",
    "eclipse light — midday but the colour is completely wrong",
    "forest midday — the canopy making everything green and filtered",
    "fog — visibility 30 metres, the world ending gently",
    "dusk in a city — the skyline still lit by the sky, the streets already lit by neon",
    "4pm in November — an hour of that specific grey that has weight to it",
]

# ══════════════════════════════════════════════════════════════════════════════
#  NSFW POOLS
# ══════════════════════════════════════════════════════════════════════════════

ETHNICITY = [
    "Russian", "Ukrainian", "Slavic", "Czech", "Polish", "German", "Nordic",
    "Scandinavian", "Swedish", "Norwegian", "Danish", "Icelandic", "Scottish",
    "English", "Irish", "French", "Italian", "Spanish", "Portuguese",
    "Austrian", "Belgian", "Dutch", "Swiss", "Croatian", "Serbian",
    "Bulgarian", "Romanian", "Hungarian", "Estonian", "Finnish",
    "Japanese", "Korean", "Chinese", "Taiwanese", "Vietnamese", "Thai",
    "Filipino", "Malaysian", "Indonesian", "Singaporean", "Cambodian",
    "Burmese", "Mongolian", "Tibetan", "Eurasian",
    "Mexican", "Brazilian", "Colombian", "Argentinian", "Cuban",
    "Puerto Rican", "Venezuelan", "Chilean", "Peruvian", "Bolivian",
    "Guatemalan", "Honduran", "Salvadoran", "Ecuadorian", "Mestiza", "Hispanic",
    "Iranian", "Indian", "Pakistani", "Bangladeshi", "Nepalese",
    "Sri Lankan", "Afghan", "Arabic", "Lebanese", "Egyptian", "Moroccan",
    "Algerian", "Turkish", "Iraqi", "Syrian",
    "Nigerian", "Kenyan", "Ghanaian", "South African", "Ethiopian",
    "Sudanese", "Somali", "Jamaican", "Haitian", "African-American",
    "Nubian", "Cameroonian",
    "Greek", "Albanian", "Armenian", "Georgian", "Macedonian", "Cypriot",
    "Maltese", "Persian",
    "African-European", "Afro-Latina", "Hapa", "Indo-African",
    "Eurasian mixed heritage", "multiracial",
    "Native Hawaiian", "Polynesian", "Samoan", "Maori",
]

BODY_TYPE = [
    "petite and slim with a flat stomach and small perky breasts",
    "curvy hourglass figure with large natural breasts and wide hips",
    "tall and lean with long legs, small breasts, and a toned stomach",
    "thick and busty with heavy breasts, a soft belly, and full thighs",
    "athletic and muscular with defined abs, firm breasts, and a tight ass",
    "pear-shaped with a narrow waist, wide hips, and a full round ass",
    "plus-size with heavy breasts, a soft round belly, and thick thighs",
    "petite with an enormous ass disproportionate to her small frame",
    "willowy and tall with small breasts, sharp collarbones, and long arms",
    "compact and stacked — short, heavy breasts, thick waist, round thighs",
    "slim waist with large implanted breasts and a surgically enhanced ass",
    "natural and full-bodied — soft everywhere, heavy breasts, round hips",
    "extremely busty on a slim frame — her breasts dominate her silhouette",
    "boyish frame with tiny breasts, narrow hips, and a tight flat ass",
    "an athletic build with a toned body",
    "a full-figured voluptuous silhouette",
    "a classic hourglass shape",
    "a chiseled muscular build with visible muscle definition",
]

AGE_RANGE = [
    "in her early twenties", "in her mid twenties", "in her late twenties",
    "in her early thirties", "in her mid thirties", "in her late thirties",
    "in her early forties", "in her mid forties",
]

HAIR_COLOR = [
    "platinum blonde", "dirty blonde", "honey blonde", "strawberry blonde",
    "warm blonde", "ash blonde", "white blonde",
    "auburn", "copper red", "fiery red", "dark red", "crimson",
    "jet black", "blue-black", "dark brown", "chestnut brown",
    "ash brown", "light brown", "hazel", "ginger",
    "silver-grey", "steel grey", "dyed purple", "dyed pink", "dyed teal",
    "dyed electric blue", "dyed neon green", "dyed magenta", "dyed lavender",
    "bleached white", "split dye — black and blonde",
    "ombre blonde tips on dark roots",
]

HAIR_STYLE = [
    "long straight hair falling past her shoulders",
    "long wavy hair loose around her face",
    "tight curls cut short",
    "loose beach waves to the middle of her back",
    "a high ponytail pulled tight",
    "hair up in a messy bun with loose strands",
    "a sharp blunt bob at jaw level",
    "box braids falling past her chest",
    "natural afro, full and round",
    "a sleek high bun", "twin pigtails",
    "shaved on the sides with long hair on top",
    "cornrows pulled back", "a long braid over one shoulder",
    "wild and dishevelled as if mid-action", "a short pixie cut",
    "a wolf cut shag with layers",
]

SKIN_TONE = [
    "snow-white porcelain skin", "pale alabaster skin",
    "fair ivory skin with light freckles", "pale skin with a rosy flush",
    "warm olive skin with a Mediterranean glow",
    "golden tan skin, sun-kissed and smooth",
    "light brown skin with warm undertones",
    "honey-colored skin with golden highlights",
    "bronze-toned skin", "amber skin", "caramel-colored skin",
    "cinnamon-hued skin", "chestnut-brown skin",
    "medium brown skin, rich and even",
    "deep brown skin with a subtle sheen",
    "deep ebony skin, velvety and dark",
    "mahogany skin tone", "mocha skin with warm undertones",
    "East Asian skin — pale with cool pink undertones",
    "South Asian skin — warm golden-brown with amber tones",
    "Latina skin — warm medium brown, smooth and even",
]

CLOTHING_STATE = [
    "completely naked",
    "completely naked except for high heels still on",
    "wearing only a thin white shirt with nothing underneath, unbuttoned",
    "in just a thong, nothing else",
    "in lingerie — a lace bra unclasped but still on, matching thong pulled to the side",
    "wearing an oversized shirt with no bottoms, riding up",
    "in a dress pulled up around her waist",
    "topless in jeans unbuttoned and falling open",
    "wearing only thigh-high stockings and nothing else",
    "in a sheer bodysuit that hides nothing",
    "naked except for jewellery — a necklace, rings",
    "in a wet shirt, soaked through and transparent",
    "half-undressed mid-strip — one strap off, zipper halfway down",
    "wearing only an open robe, untied and falling from her shoulders",
    "in athletic wear — sports bra and leggings pulled down",
    "wearing a latex bodysuit unzipped to the navel",
    "in a tiny string bikini that barely covers anything",
    "in fishnets and a garter belt, nothing else",
    "wrapped in a towel that is slipping",
    "in a BDSM harness outfit over bare skin",
    "in a gothic lace bodysuit with a leather harness over it",
]

NSFW_SETTING = [
    "on a king-size bed with white linen sheets",
    "on a leather sofa in a dimly lit living room",
    "in a hotel room — generic, anonymous, warm light",
    "in a luxury penthouse with floor-to-ceiling windows and city view",
    "on a kitchen counter, late at night",
    "on an office desk, after hours",
    "on a fur rug in front of a fireplace",
    "against a bathroom mirror, steamed up from the shower",
    "in a luxury marble spa bathroom with freestanding tub",
    "in a walk-in rain shower with glass walls",
    "on a rooftop terrace at night, city below",
    "on a yacht deck at sunset over open ocean",
    "on a secluded tropical beach at dusk",
    "in an alleyway at night, neon signs reflecting in puddles",
    "in a cheap motel room with neon light through the blinds",
    "in a Seoul apartment at night with the city glowing outside",
    "in a retro American diner at midnight",
    "in a dimly lit dungeon with chains on the wall and leather furniture",
    "in a lavish boudoir with a canopied daybed and vanity mirror",
    "in an adult photography studio with professional lighting",
    "in a private VIP booth at a strip club",
]

NSFW_MOOD = [
    "intensely focused on a single sensation",
    "completely uninhibited and unself-conscious",
    "quiet desperation building to something",
    "playful and teasing — everything is a game",
    "dominant and utterly certain",
    "submissive and completely surrendered",
    "desperate — this has been wanted for too long",
    "luxuriating — nothing else exists",
    "a proud dominant stance and gaze",
    "afterglow expression — satisfied and soft",
]

SOLO_ACTS = [
    "masturbating with her fingers, close to orgasm",
    "using a vibrator pressed against her clit",
    "fucking herself with a large dildo",
    "edging herself — stopping just before coming, again and again",
    "fingering herself slowly while watching something off-camera",
    "riding a suction-cup dildo mounted to the floor",
    "using a wand vibrator while biting her lip to stay quiet",
    "using dual toys — vibrator on her clit and dildo inside her",
    "grinding against a pillow with desperate urgency",
    "watching herself in a mirror as she fingers herself",
]

ORAL_ACTS = [
    "giving a slow, eye-contact blowjob",
    "deep throating, tears streaming, mascara running",
    "licking and stroking, working a cock with both hands",
    "gagging on a cock held at the back of her throat",
    "giving a sloppy wet blowjob, saliva on her chin",
    "sucking with her cheeks hollowed, messy and enthusiastic",
    "receiving oral — a head buried between her legs, her thighs clamped",
    "sitting on a face, grinding slowly",
    "sixty-nine position, both giving and receiving",
    "deepthroating with her hands tied behind her back",
]

PENETRATION_ACTS = [
    "being fucked missionary — legs pinned back, taking it hard",
    "riding on top, hands on a chest, bouncing fast",
    "being taken from behind doggy style, ass out",
    "being fucked against a wall, lifted off the ground",
    "bent over a surface — desk, counter, bed — taken from behind",
    "on her side, a body curled behind her fucking her slowly",
    "reverse cowgirl — facing away, grinding down hard",
    "being railed prone bone — flat on her stomach, pinned",
    "taking two at once — double penetration",
    "being fucked in the ass slowly, hands gripping the sheets",
    "riding a cock while another is in her mouth",
]

KINK_ACTS = [
    "tied up with rope — wrists above her head, ankles bound",
    "blindfolded and handcuffed, not knowing what comes next",
    "being spanked — red marks already visible on her ass",
    "wearing a collar and leash, on all fours",
    "receiving wax dripped across her breasts",
    "in a spreader bar, completely exposed",
    "being edged with a vibrator held by someone else — denied orgasm repeatedly",
    "being choked lightly during sex, her eyes wide",
    "in full brat mode — fighting back before being pinned down",
    "hogtied with rope, completely immobilised",
    "suspended from ceiling ropes in a shibari tie",
    "cuffed to a bed, unable to move, teased relentlessly",
]

EXHIBITIONIST_ACTS = [
    "masturbating in a car in a public car park",
    "flashing her breasts from a hotel balcony",
    "fucking in a changing room with the curtain not quite closed",
    "fingering herself at a restaurant under the table",
    "being fucked quietly against a wall at a party",
    "naked in front of an open window facing the street",
    "fucking on a beach just out of sight of other people",
]

GROUP_ACTS = [
    "at the centre of a gangbang, being passed between multiple men",
    "in a threesome — two men attending to her completely",
    "in an MMF — being fucked while sucking another cock",
    "in a FFM — making out with another woman while being fucked",
    "the centre of attention at a sex party — everyone's turn",
]

NSFW_ACT_POOLS = {
    "solo":          (SOLO_ACTS,          18),
    "oral":          (ORAL_ACTS,          22),
    "penetration":   (PENETRATION_ACTS,   28),
    "kink":          (KINK_ACTS,          18),
    "exhibitionist": (EXHIBITIONIST_ACTS,  7),
    "group":         (GROUP_ACTS,          7),
}

NSFW_LIGHTING = [
    None, None,
    "warm amber bedside lamp as the only light source",
    "neon light from outside casting colored stripes through blinds",
    "soft window light at golden hour",
    "candlelight only — flickering warm shadows",
    "blue moonlight through sheer curtains",
    "red emergency lighting casting deep shadows",
    "diffused soft-box studio lighting",
    "dramatic Rembrandt side lighting",
    "backlit silhouette against bright window",
    "chiaroscuro — deep shadow and bright highlight",
    "neon rim light in magenta and cyan",
]

NSFW_POSE = [
    None, None,
    "arching her back, one hand in her hair",
    "lying on her side, leg raised",
    "on all fours, looking back over her shoulder",
    "sitting with her legs slightly apart, leaning forward",
    "kneeling upright with her hands behind her head",
    "lying on her back with her legs in the air",
    "standing with her weight on one hip, shoulders back",
    "bent over, hands braced on a surface",
    "straddling something, hips forward",
    "crouching low, looking up at the camera",
    "standing with her back to camera, looking over her shoulder",
]


# ══════════════════════════════════════════════════════════════════════════════
#  ENERGY MODIFIERS
# ══════════════════════════════════════════════════════════════════════════════

def _energy_line(energy: str, content_type: str) -> str:
    if energy == "Extreme":
        if content_type == "nsfw":
            return "Push every physical and explicit element to its absolute maximum. No restraint anywhere."
        elif content_type in ("vehicle", "sports"):
            return "Maximum velocity, maximum danger, maximum noise. Nothing held back. Every sense overwhelmed."
        elif content_type in ("fantasy", "scifi", "horror"):
            return "The stakes are absolute. The scale is maximum. This is the moment everything was building toward."
        elif content_type == "animation":
            return "Cartoon chaos at maximum volume. Reactions exaggerated to breaking point. Physics optional."
        elif content_type == "crime":
            return "Maximum tension. Someone is not making it out. The moral cost is everything."
        elif content_type == "weather":
            return "The most dangerous version of this storm. No safety margin. The physics are absolute."
        elif content_type == "dialogue_scene":
            return "Every line lands like a blow. Nothing is held back. This is the conversation that ends something."
        elif content_type == "urban":
            return "The city at its most alive, most chaotic, most human. Every frame is saturated with life."
        else:
            return "Maximum intensity. Every element pushed to its limit. Nothing cautious anywhere."
    elif energy == "Fun":
        if content_type == "nsfw":
            return "Keep it light and flirty. Laughing during the scene is correct. Playful, not intense."
        elif content_type == "animation":
            return "Everything is slightly ridiculous and everyone is fine with it. Warm and loud."
        elif content_type == "food":
            return "Joyful. Food is pleasure. The cooking is the story and the story is good."
        elif content_type == "crime":
            return "Caper energy — nothing goes right and everyone is having a terrible time and it's somehow funny."
        elif content_type == "dialogue_scene":
            return "The conversation has warmth. There may be laughter. The stakes are real but the tone is light."
        elif content_type == "urban":
            return "The city as playground. Everyone is in on something. The night is young and so is everyone."
        else:
            return "Light, warm, loose. The moment feels easy and enjoyable. No heavy stakes."
    else:
        if content_type == "nsfw":
            return "Cinematic weight — every detail deliberate, grounded, and specific."
        elif content_type in ("vehicle", "sports"):
            return "Precision and focus. Every frame earns its place. The craft is visible."
        elif content_type in ("fantasy", "scifi"):
            return "World-building that earns its atmosphere. The detail makes it real."
        elif content_type == "horror":
            return "Slow and deliberate. The dread is architectural. Nothing wasted."
        elif content_type == "crime":
            return "Every word and action has weight. The moral arithmetic is exact."
        elif content_type == "weather":
            return "Awe and danger in equal measure. The scale is real and so is the human in it."
        elif content_type == "dialogue_scene":
            return "The scene breathes. Every pause is intentional. What's not said is as important as what is."
        elif content_type == "urban":
            return "Specific and observed — the city rendered in exact human detail."
        else:
            return "Grounded and specific. Every detail is chosen. The atmosphere is the story."


# ══════════════════════════════════════════════════════════════════════════════
#  BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _anchor(instruction):
    if instruction and instruction.strip():
        return f"User anchors: {instruction.strip()} — honour this, expand everything else."
    return None


def _build_vehicle(rng, instruction, energy):
    low = instruction.lower() if instruction else ""
    if any(w in low for w in ["motorcycle", "motorbike", "bike"]):
        vehicle = rng.choice(MOTORCYCLES)
    elif any(w in low for w in ["plane", "jet", "aircraft", "fly", "pilot"]):
        vehicle = rng.choice(AIRCRAFT)
    elif any(w in low for w in ["boat", "ship", "yacht", "sail", "submarine"]):
        vehicle = rng.choice(WATERCRAFT)
    elif any(w in low for w in ["race", "racing", "f1", "formula", "nascar", "rally", "drift"]):
        vehicle = rng.choice(CARS_RACING)
    elif any(w in low for w in ["street", "city", "cruise", "drive"]):
        vehicle = rng.choice(CARS_STREET)
    else:
        vehicle = rng.choice(CARS_RACING + CARS_STREET + MOTORCYCLES)
    lines = [
        f"Vehicle: {vehicle}.",
        f"Setting: {rng.choice(VEHICLE_SETTINGS)}.",
        f"Camera: {rng.choice(VEHICLE_CAMERA)}.",
    ]
    s = rng.choice(UNIVERSAL_VISUAL_STYLE)
    t = rng.choice(UNIVERSAL_TIME)
    if s: lines.append(f"Visual style: {s}.")
    if t: lines.append(f"Time/light: {t}.")
    a = _anchor(instruction)
    if a: lines.append(a)
    lines.append(_energy_line(energy, "vehicle"))
    return "\n".join(lines)


def _build_sports(rng, instruction, energy):
    lines = [
        f"Scenario: {rng.choice(SPORTS_SCENARIOS)}.",
        f"Setting: {rng.choice(SPORTS_SETTINGS)}.",
        f"Camera: {rng.choice(UNIVERSAL_CAMERA)}.",
    ]
    s = rng.choice(UNIVERSAL_VISUAL_STYLE)
    t = rng.choice(UNIVERSAL_TIME)
    if s: lines.append(f"Visual style: {s}.")
    if t: lines.append(f"Time/light: {t}.")
    a = _anchor(instruction)
    if a: lines.append(a)
    lines.append(_energy_line(energy, "sports"))
    return "\n".join(lines)


def _build_animation(rng, instruction, energy):
    lines = [
        f"World: {rng.choice(ANIMATION_WORLDS)}.",
        f"Scenario: {rng.choice(ANIMATION_SCENARIOS)}.",
        f"Tone: {rng.choice(ANIMATION_TONE)}.",
        f"Camera: {rng.choice(UNIVERSAL_CAMERA)}.",
    ]
    s = rng.choice(UNIVERSAL_VISUAL_STYLE)
    if s: lines.append(f"Visual style: {s}.")
    a = _anchor(instruction)
    if a: lines.append(a)
    lines.append(_energy_line(energy, "animation"))
    return "\n".join(lines)


def _build_fantasy(rng, instruction, energy):
    lines = [
        f"World: {rng.choice(FANTASY_WORLDS)}.",
        f"Scenario: {rng.choice(FANTASY_SCENARIOS)}.",
        f"Atmosphere: {rng.choice(FANTASY_ATMOSPHERE)}.",
        f"Camera: {rng.choice(UNIVERSAL_CAMERA)}.",
    ]
    s = rng.choice(UNIVERSAL_VISUAL_STYLE)
    t = rng.choice(UNIVERSAL_TIME)
    if s: lines.append(f"Visual style: {s}.")
    if t: lines.append(f"Time/light: {t}.")
    a = _anchor(instruction)
    if a: lines.append(a)
    lines.append(_energy_line(energy, "fantasy"))
    return "\n".join(lines)


def _build_scifi(rng, instruction, energy):
    lines = [
        f"Setting: {rng.choice(SCIFI_SETTINGS)}.",
        f"Scenario: {rng.choice(SCIFI_SCENARIOS)}.",
        f"Tone: {rng.choice(SCIFI_TONE)}.",
        f"Camera: {rng.choice(UNIVERSAL_CAMERA)}.",
    ]
    s = rng.choice(UNIVERSAL_VISUAL_STYLE)
    if s: lines.append(f"Visual style: {s}.")
    a = _anchor(instruction)
    if a: lines.append(a)
    lines.append(_energy_line(energy, "scifi"))
    return "\n".join(lines)


def _build_animal(rng, instruction, energy):
    lines = [
        f"Scenario: {rng.choice(ANIMAL_SCENARIOS)}.",
        f"Setting: {rng.choice(ANIMAL_SETTINGS)}.",
        f"Camera: {rng.choice(UNIVERSAL_CAMERA)}.",
    ]
    s = rng.choice(UNIVERSAL_VISUAL_STYLE)
    t = rng.choice(UNIVERSAL_TIME)
    if s: lines.append(f"Visual style: {s}.")
    if t: lines.append(f"Time/light: {t}.")
    a = _anchor(instruction)
    if a: lines.append(a)
    lines.append(_energy_line(energy, "animal"))
    return "\n".join(lines)


def _build_food(rng, instruction, energy):
    lines = [
        f"Scenario: {rng.choice(FOOD_SCENARIOS)}.",
        f"Setting: {rng.choice(FOOD_SETTINGS)}.",
        f"Camera: {rng.choice(UNIVERSAL_CAMERA)}.",
    ]
    s = rng.choice(UNIVERSAL_VISUAL_STYLE)
    t = rng.choice(UNIVERSAL_TIME)
    if s: lines.append(f"Visual style: {s}.")
    if t: lines.append(f"Time/light: {t}.")
    a = _anchor(instruction)
    if a: lines.append(a)
    lines.append(_energy_line(energy, "food"))
    return "\n".join(lines)


def _build_horror(rng, instruction, energy):
    lines = [
        f"Setting: {rng.choice(HORROR_SETTINGS)}.",
        f"Scenario: {rng.choice(HORROR_SCENARIOS)}.",
        f"Tone: {rng.choice(HORROR_TONE)}.",
        f"Camera: {rng.choice(UNIVERSAL_CAMERA)}.",
    ]
    s = rng.choice(UNIVERSAL_VISUAL_STYLE)
    if s: lines.append(f"Visual style: {s}.")
    a = _anchor(instruction)
    if a: lines.append(a)
    lines.append(_energy_line(energy, "horror"))
    return "\n".join(lines)


def _build_nature(rng, instruction, energy):
    lines = [
        f"Scenario: {rng.choice(NATURE_SCENARIOS)}.",
        f"Setting: {rng.choice(NATURE_SETTINGS)}.",
        f"Camera: {rng.choice(UNIVERSAL_CAMERA)}.",
    ]
    s = rng.choice(UNIVERSAL_VISUAL_STYLE)
    t = rng.choice(UNIVERSAL_TIME)
    if s: lines.append(f"Visual style: {s}.")
    if t: lines.append(f"Time/light: {t}.")
    a = _anchor(instruction)
    if a: lines.append(a)
    lines.append(_energy_line(energy, "nature"))
    return "\n".join(lines)


def _build_music(rng, instruction, energy):
    lines = [
        f"Scenario: {rng.choice(MUSIC_SCENARIOS)}.",
        f"Setting: {rng.choice(MUSIC_SETTINGS)}.",
        f"Camera: {rng.choice(UNIVERSAL_CAMERA)}.",
    ]
    s = rng.choice(UNIVERSAL_VISUAL_STYLE)
    t = rng.choice(UNIVERSAL_TIME)
    if s: lines.append(f"Visual style: {s}.")
    if t: lines.append(f"Time/light: {t}.")
    a = _anchor(instruction)
    if a: lines.append(a)
    lines.append(_energy_line(energy, "music"))
    return "\n".join(lines)


def _build_historical(rng, instruction, energy):
    lines = [
        f"Scenario: {rng.choice(HISTORICAL_SCENARIOS)}.",
        f"Setting: {rng.choice(HISTORICAL_SETTINGS)}.",
        f"Camera: {rng.choice(UNIVERSAL_CAMERA)}.",
    ]
    s = rng.choice(UNIVERSAL_VISUAL_STYLE)
    t = rng.choice(UNIVERSAL_TIME)
    if s: lines.append(f"Visual style: {s}.")
    if t: lines.append(f"Time/light: {t}.")
    a = _anchor(instruction)
    if a: lines.append(a)
    lines.append(_energy_line(energy, "historical"))
    return "\n".join(lines)


def _build_sfw_person(rng, instruction, energy):
    lines = [
        f"Destination/context: {rng.choice(SFW_DESTINATIONS)}.",
        f"Scenario: {rng.choice(SFW_PERSON_SCENARIOS)}.",
        f"Subject mood: {rng.choice(SFW_MOODS)}.",
        f"Camera: {rng.choice(UNIVERSAL_CAMERA)}.",
    ]
    s = rng.choice(UNIVERSAL_VISUAL_STYLE)
    t = rng.choice(UNIVERSAL_TIME)
    if s: lines.append(f"Visual style: {s}.")
    if t: lines.append(f"Time/light: {t}.")
    a = _anchor(instruction)
    if a: lines.append(a)
    lines.append(_energy_line(energy, "sfw_person"))
    return "\n".join(lines)


def _build_nsfw(rng, instruction, energy):
    categories = list(NSFW_ACT_POOLS.keys())
    weights    = [NSFW_ACT_POOLS[c][1] for c in categories]
    category   = rng.choices(categories, weights=weights, k=1)[0]
    act        = rng.choice(NSFW_ACT_POOLS[category][0])
    lines = [
        f"A {rng.choice(ETHNICITY)} woman {rng.choice(AGE_RANGE)}.",
        f"Body: {rng.choice(BODY_TYPE)}.",
        f"Skin: {rng.choice(SKIN_TONE)}.",
        f"Hair: {rng.choice(HAIR_COLOR)}, {rng.choice(HAIR_STYLE)}.",
        f"Clothing: {rng.choice(CLOTHING_STATE)}.",
        f"Setting: {rng.choice(NSFW_SETTING)}.",
        f"Act: {act}.",
        f"Expression/mood: {rng.choice(NSFW_MOOD)}.",
    ]
    pose = rng.choice(NSFW_POSE)
    light = rng.choice(NSFW_LIGHTING)
    style = rng.choice(UNIVERSAL_VISUAL_STYLE)
    if pose:  lines.append(f"Pose: {pose}.")
    if light: lines.append(f"Lighting: {light}.")
    if style: lines.append(f"Visual style: {style}.")
    a = _anchor(instruction)
    if a: lines.append(a)
    lines.append(_energy_line(energy, "nsfw"))
    return "\n".join(lines)


def _build_dialogue_scene(rng, instruction, energy):
    lines = [
        f"Context: {rng.choice(DIALOGUE_CONTEXTS)}.",
        f"Opening line / tone: {rng.choice(DIALOGUE_OPENERS)}",
        f"Subtext: {rng.choice(DIALOGUE_SUBTEXT)}.",
        f"Delivery: {rng.choice(DIALOGUE_DELIVERY)}.",
        f"Camera: {rng.choice(UNIVERSAL_CAMERA)}.",
    ]
    s = rng.choice(UNIVERSAL_VISUAL_STYLE)
    t = rng.choice(UNIVERSAL_TIME)
    if s: lines.append(f"Visual style: {s}.")
    if t: lines.append(f"Time/light: {t}.")
    a = _anchor(instruction)
    if a: lines.append(a)
    lines.append("Write actual spoken dialogue into the scene. The characters must say real lines. The LLM should compose 2–4 lines of spoken exchange that fit the subtext and delivery note above.")
    lines.append(_energy_line(energy, "sfw_person"))
    return "\n".join(lines)


def _build_crime(rng, instruction, energy):
    lines = [
        f"Scenario: {rng.choice(CRIME_SCENARIOS)}.",
        f"Setting: {rng.choice(CRIME_SETTINGS)}.",
        f"Tone: {rng.choice(CRIME_TONE)}.",
        f"Camera: {rng.choice(UNIVERSAL_CAMERA)}.",
    ]
    s = rng.choice(UNIVERSAL_VISUAL_STYLE)
    t = rng.choice(UNIVERSAL_TIME)
    if s: lines.append(f"Visual style: {s}.")
    if t: lines.append(f"Time/light: {t}.")
    a = _anchor(instruction)
    if a: lines.append(a)
    lines.append(_energy_line(energy, "crime"))
    return "\n".join(lines)


def _build_weather(rng, instruction, energy):
    lines = [
        f"Scenario: {rng.choice(WEATHER_SCENARIOS)}.",
        f"Setting: {rng.choice(WEATHER_SETTINGS)}.",
        f"Atmosphere: {rng.choice(WEATHER_ATMOSPHERE)}.",
        f"Camera: {rng.choice(UNIVERSAL_CAMERA)}.",
    ]
    s = rng.choice(UNIVERSAL_VISUAL_STYLE)
    t = rng.choice(UNIVERSAL_TIME)
    if s: lines.append(f"Visual style: {s}.")
    if t: lines.append(f"Time/light: {t}.")
    a = _anchor(instruction)
    if a: lines.append(a)
    lines.append(_energy_line(energy, "weather"))
    return "\n".join(lines)


def _build_urban(rng, instruction, energy):
    lines = [
        f"Scenario: {rng.choice(URBAN_SCENARIOS)}.",
        f"Setting: {rng.choice(URBAN_SETTINGS)}.",
        f"Energy: {rng.choice(URBAN_ENERGY)}.",
        f"Camera: {rng.choice(UNIVERSAL_CAMERA)}.",
    ]
    s = rng.choice(UNIVERSAL_VISUAL_STYLE)
    t = rng.choice(UNIVERSAL_TIME)
    if s: lines.append(f"Visual style: {s}.")
    if t: lines.append(f"Time/light: {t}.")
    a = _anchor(instruction)
    if a: lines.append(a)
    lines.append(_energy_line(energy, "urban"))
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

_BUILDERS = {
    "vehicle":        _build_vehicle,
    "sports":         _build_sports,
    "animation":      _build_animation,
    "fantasy":        _build_fantasy,
    "scifi":          _build_scifi,
    "animal":         _build_animal,
    "food":           _build_food,
    "horror":         _build_horror,
    "nature":         _build_nature,
    "music":          _build_music,
    "historical":     _build_historical,
    "sfw_person":     _build_sfw_person,
    "dialogue_scene": _build_dialogue_scene,
    "crime":          _build_crime,
    "weather":        _build_weather,
    "urban":          _build_urban,
    "nsfw":           _build_nsfw,
}


# SFW-only universe pool — everything except nsfw
CHAOS_UNIVERSES_SFW = [
    "vehicle", "sports", "animation", "fantasy", "scifi",
    "animal", "food", "horror", "nature", "music",
    "historical", "sfw_person", "crime", "weather", "urban",
    "dialogue_scene",
]


def build_wildcard_injection(
    seed: int = 0,
    energy: str = "Intense",
    instruction: str = "",
    content_gate: str = "Auto",
) -> str:
    """
    Main entry point. Called by gemma4_prompt_gen.py.

    seed=0          → fully random each run
    seed>0          → reproducible output
    instruction     → user's original prompt text (detection + anchoring)
    energy          → "Fun" | "Intense" | "Extreme"
    content_gate    → "Auto" | "SFW" | "NSFW"
                      SFW  — forces SFW-only pools regardless of instruction
                      NSFW — forces nsfw pool regardless of instruction
                      Auto — detect from instruction (original behaviour)

    Returns a formatted instruction block for the LLM to expand into a video prompt.
    The [WILDCARD: TYPE] header tells the LLM which universe it's working in.
    """
    rng = random.Random(seed if seed != 0 else None)

    if content_gate == "NSFW":
        # Hard-force nsfw pool, ignore instruction content
        content_type = "nsfw"
    elif content_gate == "SFW":
        # Force SFW: detect normally but override if nsfw came back
        content_type = _detect_content_type(instruction)
        if content_type in ("nsfw", "chaos"):
            content_type = rng.choice(CHAOS_UNIVERSES_SFW)
        # Also block nsfw even if instruction somehow snuck through
        if content_type == "nsfw":
            content_type = "sfw_person"
    else:
        # Auto — original behaviour
        content_type = _detect_content_type(instruction)
        if content_type == "chaos":
            content_type = rng.choice(CHAOS_UNIVERSES)

    builder = _BUILDERS.get(content_type, _build_sfw_person)
    result  = builder(rng, instruction, energy)
    return f"[WILDCARD: {content_type.upper()}]\n{result}"
