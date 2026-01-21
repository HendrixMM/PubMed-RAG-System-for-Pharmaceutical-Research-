# PubMed RAG System for Pharmaceutical Research

An AI-powered system that retrieves and validates medical literature from PubMed to answer pharmaceutical questions with evidence-based responses. 

## Overview

This project addresses a critical challenge in pharmaceutical research: ensuring that answers to drug safety and efficacy questions are supported by high-quality, relevant scientific evidence. The system retrieves research articles from PubMed using parallel search strategies and validates answer quality through multi-stage source scoring.

## Key Features

- **Multi-Strategy Retrieval**: Queries PubMed database using parallel search approaches (MeSH term mapping, keyword expansion, field-specific searches) to ensure comprehensive literature coverage
- **Quality Validation**: Two-stage validation framework that filters low-quality sources and flags unsupported claims
- **Iterative Refinement**: Automatically detects insufficient evidence and adjusts search parameters to improve source coverage
- **Source Tracking**: All medical claims link to specific PubMed sources for downstream verification
- **Safety Guardrails**: Built-in checks to ensure responsible handling of medical information

## Technical Stack

- **Language**: Python
- **API**: PubMed/NCBI E-utilities
- **Core Technologies**: 
  - REST API integration
  - Parallel query execution
  - RAG (Retrieval-Augmented Generation) architecture
  - Multi-model routing for query optimization

## System Architecture

```
User Query
    ↓
Multi-Strategy Search
    ├─ MeSH Term Mapping
    ├─ Keyword Expansion
    └─ Field-Specific Queries
    ↓
Source Relevance Scoring
    ↓
Answer Generation
    ↓
Quality Validation
    ├─ Evidence Support Check
    └─ Claim Verification
    ↓
Response with Citations
```

## Use Cases

- **Pharmaceutical Safety Questions**: Query drug interactions, side effects, and contraindications with evidence from peer-reviewed literature
- **Evidence Synthesis**: Aggregate findings across multiple research papers on specific compounds or conditions
- **Literature Review Support**: Identify relevant studies for drug discovery research

## Current Status

**Status**: Active Development (July 2024 - Ongoing)

**Recent Focus**:
- Improving source quality assessment
- Balancing retrieval comprehensiveness with answer accuracy
- Exploring extensions to neuroimaging and drug discovery literature

## Future Directions

Interested in extending validation approaches to multi-modal drug discovery data, including:
- Integration of imaging data validation
- Behavioral study assessment
- Molecular biomarker literature synthesis

## About

This project was developed as an independent learning experience to understand:
- How to work with biomedical databases and literature
- The importance of rigorous validation in medical applications
- Multi-source data integration challenges
- Evidence quality assessment in research contexts

Built by **Hendrix Majumdar-Moreau**, Computer Engineering student at Concordia University, as part of exploring computational approaches to translational neuroscience and drug discovery research.

## Contact

For questions or collaboration opportunities:
- **Email**: hendrix.majumdar-moreau@mail.concordia.ca
- **LinkedIn**: [linkedin.com/in/hendrixmm](https://linkedin.com/in/hendrixmm)
- **GitHub**: [github.com/HendrixMM](https://github.com/HendrixMM)

---

**Note**: This system is designed for research and educational purposes. Medical information should always be verified with qualified healthcare professionals.
