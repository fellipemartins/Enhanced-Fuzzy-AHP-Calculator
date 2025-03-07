# Level-ajusted enhanced Fuzzy AHP calculalor

This is a Streamlit-based FAHP application following this structure/logic:

### 1. User Inputs
- Define the number of respondents.
- Enter each respondent’s hierarchical level (Strategic, Tactical, Operational).
- Define up to 10 criteria.
- Define up to 10 alternatives.
- Input the pairwise comparisons for each respondent, using the Saaty scale adapted to TFNs.


### 2. FAHP Calculation 
- Uses Chang’s Extent Analysis according to Chang (1996 in https://doi.org/10.1016/0377-2217(95)00300-2).
- Uses the Saaty scale adapted as in Martins et al. (2023 in https://doi.org/10.1590/1678-6971/eRAMR230055.en).   
- Construct pairwise comparison matrices with triangular fuzzy numbers (TFNs).
- Compute fuzzy synthetic extent values.
- Determine priority weights as per the FAHP method.


### 3. Weighting Adjustment Based on Respondent Levels
- Strategic-Level Responses are taken as the baseline (no modification).
- Tactical-Level Adjustments:
  - +33% for TFNs below the strategic average.
  - -33% for TFNs above the strategic average.
- Operational-Level Adjustments:
  - +66% for TFNs below the strategic average.
  - -66% for TFNs above the strategic average.
- Compute the final weighted priority scores.


### 5. Streamlit Visualization
- Display fuzzy pairwise matrices.
- Show original FAHP priority weights.
- Show adjusted weights after hierarchical weighting.
- Provide an option to export results (CSV, Excel).

## How to cite this
```plain text
ABNT (NBR):
MARTINS, F. S. Level-ajusted enhanced Fuzzy AHP calculalor. Disponível em: <https://github.com/fellipemartins/Enhanced-Fuzzy-AHP-Calculator>. Aceso in: 7 mar. 2025.

APA:
Martins, F. S. (2025, Mar 07). Level-ajusted enhanced Fuzzy AHP calculalor. https://github.com/fellipemartins/python_curso.
```

Or BibTeX (LaTeX) format:
```bibtex
@misc{martins2025leveladjustedenhancedfuzzyahpcalculator,
  author = {MARTINS, Fellipe Silva},
  title = {Level-ajusted enhanced Fuzzy AHP calculalor},
  url = {https://github.com/fellipemartins/Enhanced-Fuzzy-AHP-Calculator},
  year = {2025},
  month = {March}
}
```
