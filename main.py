import streamlit as st
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import io
from utils import (
    triangular_fuzzy_saaty,
    inverse_tfn,
    calculate_fuzzy_synthetic_extent,
    adjust_weights_matrix,
    validate_matrix,
    defuzzify_tfn,
    normalize_crisp_weights
)

# Page Configuration
st.set_page_config(
    page_title="Fuzzy AHP Calculator",
    page_icon="📊",
    layout="wide"
)

def show_sidebar():
    """Display methodology explanation in sidebar."""
    st.sidebar.title("FAHP Methodology")
    st.sidebar.markdown("""
    ### Fuzzy Analytic Hierarchy Process (FAHP)
    FAHP extends the traditional AHP by incorporating fuzzy logic to handle uncertainty 
    and vagueness in decision-making.

    #### Key Components:
    1. **Pairwise Comparisons**
       - Uses Saaty's 1-9 scale
       - Values represent relative importance
       - Reciprocal values automatically calculated

    2. **Scale Interpretation**
       - 1: Equal importance
       - 3: Moderate importance
       - 5: Strong importance
       - 7: Very strong importance
       - 9: Absolute importance

    3. **Triangular Fuzzy Numbers (TFNs)**
       - Each comparison converted to fuzzy number
       - Each TFN is -1/+1, except 1 (--> 1, 1, 2) and 9 (--> 8, 9, 9)
       - Reciprocals use inverse TFN (1/u, 1/m, 1/l)
    """)

def input_section():
    """Handle user inputs for criteria and respondents."""
    st.title("Level-adjusted Fuzzy AHP Calculator")
    st.markdown("""
    This calculator implements a hierarchical Fuzzy Analytic Hierarchy Process (FAHP) 
    with improved numerical stability and advanced weighting mechanisms.

    The algorithm first uses Chang's method (1996) to calculate Fuzzy AHP.
    Then, if we have responses from non-strategic personnel (tactical, operational), a weighting scheme is applied:

    Tactical:
    - +33% for each number in the TFN under the average of each TFN numbers for strategic response
    - -33% if over

    Operational:
    - +66% for each number in the TFN under the average of each TFN numbers for strategic response
    - -66% if over

    That's it!
    """)

    col1, col2 = st.columns(2)
    with col1:
        num_criteria = st.number_input(
            "Number of Criteria",
            min_value=2,
            max_value=10,
            value=3,
            help="Enter the number of criteria (2-10)"
        )

    with col2:
        num_respondents = st.number_input(
            "Number of Respondents",
            min_value=1,
            max_value=20,
            value=3,
            help="Enter the number of respondents (1-20)"
        )

    # Criteria names input
    criteria_names = []
    st.subheader("Criteria Names")
    cols = st.columns(3)
    for i in range(num_criteria):
        col_idx = i % 3
        with cols[col_idx]:
            name = st.text_input(
                f"Criterion {i+1}",
                value=f"C{i+1}",
                key=f"criterion_{i}"
            )
            criteria_names.append(name)

    return num_criteria, num_respondents, criteria_names

def collect_respondent_data(num_respondents):
    """Collect respondent information."""
    respondents = []
    st.subheader("Respondent Information")

    cols = st.columns(2)
    for i in range(num_respondents):
        with cols[0]:
            name = st.text_input(
                f"Respondent {i+1} Name",
                key=f"name_{i}",
                value=f"Respondent {i+1}"
            )
        with cols[1]:
            level = st.selectbox(
                f"Respondent {i+1} Level",
                ["Strategic", "Tactical", "Operational"],
                key=f"level_{i}"
            )
        respondents.append((name, level))

    return respondents

def collect_comparisons(num_criteria, criteria_names, respondents):
    """Collect pairwise comparisons from respondents using a step-by-step process."""
    pairwise_matrices = {}

    for name, level in respondents:
        st.subheader(f"Pairwise Comparisons - {name} ({level})")
        matrix = np.ones((num_criteria, num_criteria, 3))

        st.info("""
        For each comparison, you will:
        1. Indicate if the criteria are equally important
        2. If not equal, select which criterion is more important
        3. Specify how much more important using the scale (1-9)
        """)

        for (c1, c2) in combinations(range(num_criteria), 2):
            st.write(f"\nComparing {criteria_names[c1]} and {criteria_names[c2]}")

            # Step 1: Equal or not
            equal = st.radio(
                f"Are {criteria_names[c1]} and {criteria_names[c2]} equally important?",
                options=["Yes", "No"],
                key=f"equal_{name}_{c1}_{c2}"
            )

            if equal == "Yes":
                # If equal, set both comparisons to 1
                matrix[c1][c2] = triangular_fuzzy_saaty(1)
                matrix[c2][c1] = triangular_fuzzy_saaty(1)
            else:
                # Step 2: Which one is more important
                more_important = st.radio(
                    "Which criterion is more important?",
                    options=[criteria_names[c1], criteria_names[c2]],
                    key=f"important_{name}_{c1}_{c2}"
                )

                # Step 3: How much more important
                if more_important == criteria_names[c1]:
                    scale = st.slider(
                        f"How much more important is {criteria_names[c1]}?",
                        min_value=2,
                        max_value=9,
                        value=2,
                        key=f"scale_{name}_{c1}_{c2}",
                        help="""
                        2-3: Slightly more important
                        4-5: More important
                        6-7: Much more important
                        8-9: Extremely more important
                        """
                    )
                    tfn = triangular_fuzzy_saaty(scale)
                    matrix[c1][c2] = tfn
                    matrix[c2][c1] = inverse_tfn(tfn)
                else:
                    scale = st.slider(
                        f"How much more important is {criteria_names[c2]}?",
                        min_value=2,
                        max_value=9,
                        value=2,
                        key=f"scale_{name}_{c2}_{c1}",
                        help="""
                        2-3: Slightly more important
                        4-5: More important
                        6-7: Much more important
                        8-9: Extremely more important
                        """
                    )
                    tfn = triangular_fuzzy_saaty(scale)
                    matrix[c2][c1] = tfn
                    matrix[c1][c2] = inverse_tfn(tfn)

            st.markdown("---")  # Visual separator between comparisons

        if not validate_matrix(matrix):
            st.error(f"Invalid comparison matrix for {name}")
            continue

        pairwise_matrices[name] = (matrix, level)

    return pairwise_matrices

def calculate_priorities(pairwise_matrices):
    """Calculate priority vectors with hierarchical adjustments."""
    # First process strategic responses
    strategic_matrices = [
        matrix for (matrix, level) in pairwise_matrices.values()
        if level == "Strategic"
    ]

    if not strategic_matrices:
        st.warning("No strategic-level respondents found. Using default weights.")
        strategic_avg = np.ones_like(next(iter(pairwise_matrices.values()))[0])
    else:
        # Calculate average matrix from strategic responses
        strategic_avg = np.mean(strategic_matrices, axis=0)

    priority_vectors = {}
    for name, (matrix, level) in pairwise_matrices.items():
        # First adjust the comparison matrix based on level
        adjusted_matrix = adjust_weights_matrix(matrix, strategic_avg, level)

        # Then calculate synthetic extent values
        weights = calculate_fuzzy_synthetic_extent(adjusted_matrix)

        # Store fuzzy components
        priority_vectors[f"{name}_l"] = weights[:, 0]  # lower bounds
        priority_vectors[f"{name}_m"] = weights[:, 1]  # middle values
        priority_vectors[f"{name}_u"] = weights[:, 2]  # upper bounds

        # Calculate and store defuzzified weights
        defuzz_weights = np.array([
            defuzzify_tfn(l, m, u) 
            for l, m, u in zip(
                weights[:, 0],
                weights[:, 1],
                weights[:, 2]
            )
        ])
        # Normalize defuzzified weights
        priority_vectors[f"{name}_crisp"] = normalize_crisp_weights(defuzz_weights)

    return priority_vectors

def calculate_final_weights(priority_vectors, criteria_names):
    """Calculate final aggregated weights across all respondents."""
    # Get all crisp weights
    crisp_columns = [col for col in priority_vectors if col.endswith('_crisp')]
    crisp_weights = pd.DataFrame(priority_vectors)[crisp_columns]

    # Calculate mean weights across respondents
    final_weights = crisp_weights.mean(axis=1)

    # Create DataFrame with criteria names
    final_df = pd.DataFrame({
        'Criterion': criteria_names,
        'Weight': final_weights
    })

    # Sort by weight in descending order
    final_df = final_df.sort_values('Weight', ascending=False)

    return final_df

def display_results(priority_vectors, criteria_names):
    """Display and visualize results."""
    st.subheader("Fuzzy Weights by Respondent")

    # Group results by respondent
    priority_vectors_df = pd.DataFrame(priority_vectors)
    for respondent in set(col.split('_')[0] for col in priority_vectors_df.columns):
        st.write(f"\n### Results for {respondent}")

        # Display fuzzy weights
        fuzzy_cols = [f"{respondent}_l", f"{respondent}_m", f"{respondent}_u"]
        fuzzy_df = priority_vectors_df[fuzzy_cols].copy()
        fuzzy_df.columns = ['Lower', 'Middle', 'Upper']
        fuzzy_df.index = criteria_names

        st.dataframe(
            fuzzy_df.style.format("{:.4f}")
            .background_gradient(cmap='YlOrRd')
        )

    # Calculate and display final aggregated weights
    st.subheader("Final Aggregated Weights")
    final_weights = calculate_final_weights(priority_vectors_df, criteria_names)

    st.write("Priority weights ordered by importance:")
    st.dataframe(
        final_weights.style.format({
            'Weight': '{:.4f}'
        }).background_gradient(
            subset=['Weight'],
            cmap='YlOrRd'
        ),
        hide_index=True
    )

    # Visualization of final weights
    st.subheader("Final Weights Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.bar(final_weights['Criterion'], final_weights['Weight'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Export options
    st.subheader("Export Results")

    # Excel Export
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write fuzzy weights
        priority_vectors_df.to_excel(writer, sheet_name='Fuzzy Weights', index=True)
        # Write final weights
        final_weights.to_excel(writer, sheet_name='Final Weights', index=False)

    st.download_button(
        "Download Results (Excel)",
        output.getvalue(),
        "fahp_results.xlsx",
        "application/vnd.ms-excel",
        key='download-excel'
    )

def main():
    """Main application flow."""
    show_sidebar()

    num_criteria, num_respondents, criteria_names = input_section()
    respondents = collect_respondent_data(num_respondents)

    if not all(name for name, _ in respondents):
        st.error("Please enter all respondent names")
        return

    pairwise_matrices = collect_comparisons(
        num_criteria,
        criteria_names,
        respondents
    )

    if st.button("Calculate Priorities"):
        with st.spinner("Calculating priorities..."):
            priority_vectors = calculate_priorities(pairwise_matrices)
            display_results(priority_vectors, criteria_names)

if __name__ == "__main__":
    main()
