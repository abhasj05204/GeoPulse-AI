# Dataset source and handling

## NYC 311 Service Requests from 2020 to Present

Primary source: [NYC Open Data dataset erm2-nwe9](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9/data). The city describes it as a daily-updated set of service requests that can be directed to specific agencies. The source page currently lists 23.2 million rows and 44 columns; fields include unique key, created date, complaint type, latitude, and longitude. The city warns that expected values can change and category value lists are not exhaustive. Verify the live schema before each formal acquisition. [Dataset page](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9/data).

The downloader samples across calendar months and within each month at early, middle, and late offsets so a bounded download is not restricted to the earliest records in the entire requested period. The download is still a sample, not a census. Record the query dates, requested limit, returned row count, and acquisition time.

## Infrastructure cohort

The default modelling cohort includes the source complaint types `Street Condition`, `Street Light Condition`, `Sidewalk Condition`, `Sewer`, `Water System`, and `Highway Condition`. These values are configurable through `GEOPULSE_INFRA_TYPES` or `--complaint-types`. Run `python scripts/inspect_data.py data/raw/nyc311.csv` first and verify the actual category values. If none match, training stops rather than silently using all 311 categories.

## Data handling

A Socrata application token is optional for small exploratory downloads but recommended for larger acquisitions. Set `SOCRATA_APP_TOKEN` locally. Raw and processed datasets are excluded from Git. Keep a source audit with row counts, date coverage, invalid coordinates, duplicate counts, and top complaint categories.

## Spatial context

NYC coordinates are projected to EPSG:32618 for metre-based grid aggregation and DBSCAN. This projection is NYC-specific. OpenStreetMap enrichment is not yet implemented; any future use must follow ODbL attribution and the selected service's fair-use policy.
