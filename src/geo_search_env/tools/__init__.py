"""The agent's tools (offline stand-ins for what a professional geolocator uses online; AGENT_DIRECTION.md).

  pinpoint   visual search: Pinpoint's retrieval + reranker over MP16, optionally restricted (box, country / region, near points, rows)
  maps       Google Maps: name search with near=, phone / website lookup, features_near (OSM + GeoNames + Overture; FeatureIndex for batches)
  web        Google web search: Wikipedia / Wikivoyage pages, plus where the words occur in Flickr titles / tags and named places

Each tool has two faces: a text one the agent reads (Maps / Google-style result lists, no hidden coordinates beyond what Google shows) and
structured batch calls for training and features (Pinpoint.search over many photos, MapsSearch.query -> Place, FeatureIndex, TextSearch.matches).
The pipeline's read chain (observe -> text_search / region_evidence -> clue_judge) stays in experiment/ and is what these tools wrap.
Data builders: tools.osm_extract (OSM planet), tools.maps build, experiment.text_search / web_search build. Archived tools (landmark matching,
GeoNames / Overture POI tools, the later clue checks, map / name checks): git tag pre-tool-consolidation.
"""
