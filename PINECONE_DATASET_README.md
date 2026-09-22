# Pinecone ICD-10-CM Dataset Builder

Builds a private Pinecone vector index from the official FY2026 ICD-10-CM code-description release published by CDC/NCHS and linked by CMS.

Official source:
https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/2026/icd10cm-code-descriptions-2026.zip

FY2026 is effective for healthcare services/encounters from October 1, 2026 through September 30, 2026.

## Build flow

Official ICD-10-CM ZIP -> parser -> hierarchy metadata -> embeddings -> Pinecone.

The builder stores metadata including code, title, parent_code, level, billable, source and effective dates.

## Setup

pip install -r requirements-pinecone.txt

Set OPENAI_API_KEY and PINECONE_API_KEY. Optional defaults are in .env.example.

## Test

python scripts/build_pinecone_index.py --dry-run

## Small test upload

python scripts/build_pinecone_index.py --limit 100

## Full build

python scripts/build_pinecone_index.py

The index defaults to amaze-icd10cm-2026 and namespace icd10cm_2026.

Do not commit API keys or downloaded ICD-10 files. This creates a separate Amaze index and does not access another project's private Pinecone database.

The vector index is a retrieval/catalog layer. Final coding should still be checked against the current official ICD-10-CM Tabular List, Alphabetic Index and Coding Guidelines.
