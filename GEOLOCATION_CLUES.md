# Clues expert geolocators use (2026-10-08)

A catalogue of what professional geolocators (OSINT analysts, top GeoGuessr players such as Rainbolt) read from an image, for designing
the agent's observe step and tools (`EXPERT_PLAN.md`). Each clue lists what it narrows down to and an example. Examples are
typical, not rules: conventions have exceptions, change over time and vary inside countries.

Resolution: **W** world half / hemisphere, **C** country (or a group of countries), **R** region / province, **L** city / locality, **P** point (street, building).

How experts use them: never one clue alone. Start broad (hemisphere, driving side, script), cut to a few countries, then a region
(plates, area codes, signs, vegetation), then search a map for a place where everything visible co-occurs, and verify against
satellite or street imagery.

## 1. Text and language

| Clue | Narrows to | Example |
|---|---|---|
| Script | C | Cyrillic, Greek, Hangul, Thai, Devanagari, Georgian, Armenian, Hebrew, Arabic |
| Language within a script | C / R | Cyrillic: Russian vs Ukrainian (і, ї, є) vs Serbian vs Bulgarian; Latin: Catalan vs Spanish |
| Diacritics and letters | C | ő / ű Hungarian, ł Polish, ø / å Nordic, ß German, ñ Spanish, ã / õ Portuguese, ț / ș Romanian |
| Spelling variants | C | colour / color, centre / center, tyre / tire |
| Bilingual or minority-language signs | R | Welsh + English (Wales), Basque, Catalan, Frisian, French + English (Canada), Swedish in Finland |
| Place names on signs, shops, vehicles | L / P | town names, street names, "Comune di Lucca" |
| Business names | P | a shop, restaurant, hotel or gallery name found in a POI database |
| Street address, house number | P | "Calle Mayor 14" |
| Postcode format and value | C / P | UK "SW1A 1AA", Netherlands "1012 AB", US ZIP |
| Phone number format and area code | C / R | +421 Slovakia; "(415)" San Francisco; "030" Berlin |
| Website domain | C / R / P | .lu Luxembourg; .bayern, .nyc, .cat; the domain of a specific business |
| Email providers | C | web.de, yandex.ru, naver.com |
| Currency, price format | C | "1.234,50 €", "£", "¥", "R$" |
| Units | C | mph vs km/h; miles on distance signs (UK, US) |
| Date and time formats | C | 08/10 vs 10/08, 24-hour clock on posters |
| Event banners, posters, election posters | L | race or festival names; candidate names give a constituency |
| Plaques, memorials, inscriptions | P | "On this site in 1842…" |
| Station names, departure boards | P | a platform sign or a destination list |
| Local newspapers, sports teams, universities | L | a newspaper box, a jersey, a campus banner |
| QR codes | P | decode to a URL |

## 2. Roads and road markings

| Clue | Narrows to | Example |
|---|---|---|
| Driving side | C | left: UK, Ireland, Japan, Australia, New Zealand, India, South Africa, Thailand, Indonesia, Kenya |
| Centre-line colour | C | yellow in most of the Americas, Norway and Finland; white in most of Europe |
| Line patterns | C | dash length and spacing; double centre lines; UK short dashes |
| Edge lines | C / R | yellow vs white edges; none at all on minor roads |
| Kerb painting | C / R | painted kerbs (e.g. black-and-white, red-and-white) mark specific countries |
| Road surface | C / R | concrete slabs, cobbles, red asphalt cycle lanes (Netherlands), dirt roads |
| Road width, shoulders, verges | C / R | wide US rural roads vs narrow UK lanes with hedges |
| Guardrails | C | rail profile and post shape differ by country |
| Kilometre / mile markers | R / P | "N-340 km 214": a road and a position on it |
| Roundabouts, junction layouts | C | very common in France and the UK; rare in much of the US |

## 3. Road signs

| Clue | Narrows to | Example |
|---|---|---|
| Direction-sign colours | C | Italy and Switzerland: green motorways, blue other roads; France and Germany: blue motorways; UK: blue motorways, green primary routes |
| Warning-sign shape | C | yellow diamond in the Americas, Australia, Japan; red-bordered triangle in Europe; yellow-background triangles in Sweden, Finland, Poland |
| Stop signs | C | STOP vs ALTO (Mexico) vs PARE (Brazil) vs a triangle (Japan) |
| Speed-limit signs | C | US rectangular "SPEED LIMIT" vs European red circle |
| Pedestrian crossing signs | C | blue square in much of Europe; yellow-green in the Americas |
| Fonts | C | Highway Gothic (US), Transport (UK), DIN (Germany) |
| Route shields and road numbers | C / R | Interstate shield, "A-7", "E45"; numbering systems differ by country |
| Direction signs with distances | P | "Toledo 12 / Madrid 70": two or three distances intersect at one spot |
| Chevrons | C | colour (black-on-white, red-on-white, yellow) by country |
| Street-name signs | C / L | blue enamel (Paris), green blades (US), the city's own design |

## 4. Street furniture and utilities

| Clue | Narrows to | Example |
|---|---|---|
| Bollards / roadside posts | C | each European country has its own design (France, Germany, Netherlands, Czechia, Poland…) |
| Utility poles | C / R | wood vs concrete vs steel; concrete poles with holes (Hungary); pole tags and numbering |
| Wires, insulators, transformers | C | overhead vs buried lines; transformer shape |
| Street lights | C / L | lamp shape; cities often have their own design |
| Traffic lights | C | horizontal (Japan, parts of the US) vs vertical; backplate colour; lights before or after the stop line |
| Pedestrian signals | C | figure style, countdown timers |
| Rail crossings | C | crossbuck and barrier designs |
| Post boxes | C | red (UK), yellow (France, Germany), blue (US) |
| Fire hydrants | C / L | colour and shape; some cities paint their own colours |
| Manhole covers | L | often cast with the city's name or crest (Japan's decorated covers) |
| Phone booths, bins, bus stops | C / L | design varies by country and city |
| House numbers | C / L | style of house-number plates |

## 5. Vehicles

| Clue | Narrows to | Example |
|---|---|---|
| Number-plate shape and colour | C | yellow rear plates (UK, Netherlands, Luxembourg, Israel); blue EU band; long vs short plates |
| Regional plate codes | R | German district codes, French département numbers, Russian region numbers, the province character on Chinese plates |
| No front plate | C / R | allowed in some US states |
| Vehicle mix | C | kei cars (Japan), Lada (Russia), tuk-tuks (South / Southeast Asia), motorbike-dominated traffic, pickup density |
| Taxis | C / L | yellow (New York), black cabs (London), colour schemes by city |
| Buses, trams, trains | L | operator liveries and names |
| Police, ambulance livery | C | "Guardia Civil", "Carabinieri", "Gendarmerie" |
| Company names on vans and trucks | L | local businesses with phone numbers |
| Boats | L | home port on the stern; fishing-port codes (e.g. "PZ", Penzance) |
| Aircraft | C | registration prefix (G- UK, D- Germany, N US) |

## 6. Architecture and the built environment

| Clue | Narrows to | Example |
|---|---|---|
| Roof shape and material | C / R | terracotta tiles (Mediterranean), slate, steep snow roofs, flat roofs (hot dry climates), curved eaves (East Asia) |
| Wall materials and colour | C / R | brick (Netherlands, UK), adobe, painted render, timber (Nordic red houses) |
| Windows and shutters | C | shutter styles; window bars |
| Rooftop water tanks, solar water heaters | C / R | Mediterranean, Middle East, parts of Latin America |
| Fences, walls, gates | C | high walls with gates (Latin America, Africa), hedges (UK) |
| Housing types | C / R | Soviet-era blocks, US suburban houses, Japanese narrow houses |
| Religious buildings | C / R | church styles (Orthodox onion domes), mosques, temples, shrines, pagodas |
| Urban layout | C / L | grid streets (US), medieval centres (Europe) |
| Skyline | L | unique high-rises and towers |

## 7. Landmarks and named places

| Clue | Narrows to | Example |
|---|---|---|
| Famous buildings and monuments | P | towers, bridges, cathedrals, stadiums |
| Lesser-known distinctive structures | P | a TV tower, a statue, a frieze in a metro station |
| Natural landmarks | P | a recognisable mountain, waterfall, rock formation |
| Viewpoints | P | the same view many photographers took |

## 8. Commerce

| Clue | Narrows to | Example |
|---|---|---|
| Regional chains | C / R | Wawa, H-E-B (US regions), Mercadona (Spain), Biedronka (Poland), Tesco vs Sainsbury's |
| Fuel-station brands | C / R | brand mix differs by country |
| Banks, telecom operators | C | national banks and mobile networks on signs |
| Advertisements | C / R | local products, brands, services |
| Markets and street vendors | C | style of stalls and goods |

## 9. Landscape, vegetation, climate

| Clue | Narrows to | Example |
|---|---|---|
| Terrain | R / P | flat plains, ridgelines, volcano shapes; a skyline of peaks can be matched to an elevation model |
| Soil colour | R | red soil (Brazil, the Australian outback, parts of Africa) |
| Rock and geology | R | limestone karst, granite, volcanic black sand |
| Coastline, water colour | R / P | fjords, coral water, a coast's shape |
| Tree species | C / R | birch and spruce (north), palms (types differ), eucalyptus (Australia, but planted in Iberia, California, Brazil, Ethiopia), cacti |
| Crops and farming | R | rice paddies, tea, vineyards, olive groves, field shapes and sizes |
| Vegetation density and season | R | lush vs arid; snow; autumn colours |
| Climate signals | W / R | haze, dust, humidity, frost |
| Animals | C / R | livestock types, camels, kangaroo warning signs |

## 10. Sun, sky and time

| Clue | Narrows to | Example |
|---|---|---|
| Sun direction | W | sun to the north at noon means the southern hemisphere (outside the tropics) |
| Sun elevation and shadow length | latitude band | with a known date and time, the sun's height gives latitude |
| Shadow direction plus a known street orientation | P | narrows a place to streets running that way |
| Day length, season | W / R | snow in July means the southern hemisphere or high altitude |

## 11. People and culture

| Clue | Narrows to | Example |
|---|---|---|
| Flags | C / R | national, regional (US states, Catalonia, Bavaria) and party flags |
| Clothing | C / R | traditional dress, school uniforms |
| Sports | C | cricket, baseball, Gaelic games |
| Religious signs | C / R | roadside shrines, prayer flags |
| Food | C / R | dishes on menus, street food |
| Festivals | L | event-specific decoration |

## 12. Imagery-specific clues (Street View only; not in our photos)

Many GeoGuessr "meta" clues come from Google's capture process, not the place: camera generation, the visible car (roof rack, a snorkel
in Kenya, an escort car), blur patterns, coverage availability, copyright year. They don't apply to Flickr photos.

## 13. Outside context (not allowed in our setup)

EXIF time and camera, reverse image search, the photographer's other photos, social-media context, watermarks added by the photographer.
Experts use them all the time; our rules ban them (contamination, same-photographer exclusion) except for a deployed agent.

## 14. Verification methods

How experts confirm a hypothesis once a few places are left:
- Search a map for features that co-occur: "church tower + river + rail bridge" near each candidate (Overpass-style queries).
- Geocode names; look up businesses, postcodes and phone numbers.
- Intersect distances from direction signs.
- Match a skyline or ridgeline to an elevation model (PeakVisor-style).
- Compute the sun position for a date and time (SunCalc-style).
- Compare satellite imagery: road layout, roof shapes, field patterns.
- Compare street-level imagery: the same building from another angle.

## What fits our photos and tools

Our targets are personal Flickr photos, not street-level panoramas. In the eye checks (lessons 35, 57, 62), most hard photos show nothing
place-specific; the ones that do mostly show text, a landmark or a business name. So, roughly by usefulness here:

| Group | Why it matters here | Data on this machine |
|---|---|---|
| Text (section 1) | Most decisive; the VLM reads it well | GeoNames (incl. postal codes); YFCC titles / tags; Overture places (names, websites, phones, brands: re-download); `phonenumbers` |
| Landmarks (section 7) | Decisive when recognised; recognition is the bottleneck (lessons 58-62) | MP16 gallery, YFCC titles / tags |
| Region-level clues: plates, area codes, regional chains, local authorities (sections 1, 5, 8) | Target the main failure: right country, wrong region | Rule tables to build; Overture brand footprints |
| Signs, roads, furniture (sections 2-4) | Mostly in street scenes, a minority of our photos | Rule tables to build; OSV-5M as labelled examples |
| Vegetation, architecture, landscape (sections 6, 9) | Already captured by visual retrieval (lesson 21) | — |
| Sun and shadows (section 10) | Needs capture time, rarely available | — |
