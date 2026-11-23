# Dataset source and handling

## NYC 311 Service Requests
The initial candidate source is NYC Open Data's Socrata service-request resource, default identifier `erm2-nwe9`. The downloader accepts a bounded date range, paginates, and requests record ID, creation time, complaint type, and coordinates. Check the current portal schema, category coverage, data dictionary, and terms before each formal run; dataset fields and portal configurations can change.

Set `NYC311_DATASET_ID` and optionally `SOCRATA_APP_TOKEN` in the local environment. Raw and processed data are excluded from Git. Record acquisition time, query range, row count, coordinate completeness, duplicates, and source metadata alongside experiment results.

## Spatial context
NYC coordinates are projected to EPSG:32618 for metre-based grid aggregation and DBSCAN. This CRS is not suitable for arbitrary global datasets. OpenStreetMap enrichment is not yet implemented; any future use must follow ODbL attribution and service fair-use rules.
