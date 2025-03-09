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
    normalize_crisp_weights,
    calculate_alternative_priorities
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
    FAHP is an extension of the AHP method that incorporates fuzzy logic to handle uncertainty in decision-making.

    #### Key Components:
    1. **Criteria Evaluation**
       - Compare criteria importance
       - Generate criteria weights

    2. **Alternative Evaluation (Optional)**
       - Each respondent can choose to evaluate alternatives
       - Compare alternatives under each criterion
       - Skip if not relevant or uncertain
       - Final rankings only include provided evaluations

    3. **Scale Interpretation**
       - 1: Equal importance
       - 3: Moderate importance
       - 5: Strong importance
       - 7: Very strong importance
       - 9: Extreme importance
       - Intermediate values can be chosen
    """)

def input_section():
    """Handle user inputs for criteria and alternatives."""
    st.title("Level-adjusted Fuzzy AHP Calculator")
    st.markdown("""
    This calculator implements a hierarchical FAHP with both criteria 
    and alternative evaluations.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        num_criteria = st.number_input(
            "Number of Criteria",
            min_value=2,
            max_value=10,
            value=3,
            help="Enter the number of criteria (2-10)"
        )

    with col2:
        num_alternatives = st.number_input(
            "Number of Alternatives",
            min_value=0,
            max_value=10,
            value=0,
            help="Enter the number of alternatives (0-10). Set to 0 to skip alternative evaluation."
        )

    with col3:
        num_respondents = st.number_input(
            "Number of Respondents",
            min_value=1,
            max_value=20,
            value=3,
            help="Enter the number of respondents (1-20)"
        )

    # Criteria names input
    st.subheader("Criteria Names")
    criteria_names = []
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

    # Alternative names input (only if alternatives > 0)
    alternative_names = []
    if num_alternatives > 0:
        st.subheader("Alternative Names")
        cols = st.columns(3)
        for i in range(num_alternatives):
            col_idx = i % 3
            with cols[col_idx]:
                name = st.text_input(
                    f"Alternative {i+1}",
                    value=f"A{i+1}",
                    key=f"alternative_{i}"
                )
                alternative_names.append(name)

    return num_criteria, num_alternatives, num_respondents, criteria_names, alternative_names

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

def collect_comparisons(num_criteria, num_alternatives, criteria_names, alternative_names, respondents):
    """Collect all pairwise comparisons with collapsible sections per respondent."""
    criteria_matrices = {}
    alternative_matrices = {cname: {} for cname in criteria_names}

    for name, level in respondents:
        with st.expander(f"Evaluations by {name} ({level})", expanded=False):
            st.info("""
            Please provide your evaluations for:
            1. Criteria comparisons (required)
            2. Alternative comparisons (optional)

            Use the scale:
            1: Equal importance
            3: Moderate importance
            5: Strong importance
            7: Very strong importance
            9: Extreme importance
            """)

            # Create tabs for criteria and alternatives
            criteria_tab, alternatives_tab = st.tabs(["Criteria Comparisons", "Alternative Comparisons"])

            # Criteria Comparisons
            with criteria_tab:
                st.subheader("Criteria Comparisons")
                matrix = np.ones((num_criteria, num_criteria, 3))

                for (c1, c2) in combinations(range(num_criteria), 2):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"\nComparing {criteria_names[c1]} and {criteria_names[c2]}")
                        equal = st.radio(
                            f"Are they equally important?",
                            options=["Yes", "No"],
                            key=f"criteria_equal_{name}_{c1}_{c2}"
                        )

                    if equal == "Yes":
                        matrix[c1][c2] = triangular_fuzzy_saaty(1)
                        matrix[c2][c1] = triangular_fuzzy_saaty(1)
                    else:
                        with col2:
                            more_important = st.radio(
                                "Which criterion is more important?",
                                options=[criteria_names[c1], criteria_names[c2]],
                                key=f"criteria_important_{name}_{c1}_{c2}"
                            )

                            scale = st.slider(
                                "How much more important?",
                                min_value=2,
                                max_value=9,
                                value=2,
                                key=f"criteria_scale_{name}_{c1}_{c2}"
                            )

                            if more_important == criteria_names[c1]:
                                tfn = triangular_fuzzy_saaty(scale)
                                matrix[c1][c2] = tfn
                                matrix[c2][c1] = inverse_tfn(tfn)
                            else:
                                tfn = triangular_fuzzy_saaty(scale)
                                matrix[c2][c1] = tfn
                                matrix[c1][c2] = inverse_tfn(tfn)

                    st.markdown("---")

                if validate_matrix(matrix):
                    criteria_matrices[name] = (matrix, level)
                else:
                    st.error(f"Invalid criteria comparison matrix")
                    return None, None

            # Alternative Comparisons
            if num_alternatives > 0:
                with alternatives_tab:
                    st.subheader("Alternative Comparisons")
                    evaluate_alternatives = st.radio(
                        "Would you like to evaluate alternatives?",
                        options=["Yes", "No Answer"],
                        key=f"evaluate_alternatives_{name}"
                    )

                    if evaluate_alternatives == "Yes":
                        for cname in criteria_names:
                            st.write(f"### Comparing Alternatives for {cname}")
                            alt_matrix = np.ones((num_alternatives, num_alternatives, 3))

                            for (a1, a2) in combinations(range(num_alternatives), 2):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"\nComparing {alternative_names[a1]} and {alternative_names[a2]}")
                                    equal = st.radio(
                                        f"Are they equally preferred?",
                                        options=["Yes", "No"],
                                        key=f"alt_equal_{cname}_{name}_{a1}_{a2}"
                                    )

                                if equal == "Yes":
                                    alt_matrix[a1][a2] = triangular_fuzzy_saaty(1)
                                    alt_matrix[a2][a1] = triangular_fuzzy_saaty(1)
                                else:
                                    with col2:
                                        more_important = st.radio(
                                            "Which alternative is better?",
                                            options=[alternative_names[a1], alternative_names[a2]],
                                            key=f"alt_important_{cname}_{name}_{a1}_{a2}"
                                        )

                                        scale = st.slider(
                                            "How much better?",
                                            min_value=2,
                                            max_value=9,
                                            value=2,
                                            key=f"alt_scale_{cname}_{name}_{a1}_{a2}"
                                        )

                                        if more_important == alternative_names[a1]:
                                            tfn = triangular_fuzzy_saaty(scale)
                                            alt_matrix[a1][a2] = tfn
                                            alt_matrix[a2][a1] = inverse_tfn(tfn)
                                        else:
                                            tfn = triangular_fuzzy_saaty(scale)
                                            alt_matrix[a2][a1] = tfn
                                            alt_matrix[a1][a2] = inverse_tfn(tfn)

                                st.markdown("---")

                            if validate_matrix(alt_matrix):
                                alternative_matrices[cname][name] = (alt_matrix, level)
                            else:
                                st.error(f"Invalid alternative comparison matrix for {cname}")
                                return None, None

            st.success(f"Successfully recorded all evaluations for {name}")

    return criteria_matrices, alternative_matrices

def calculate_final_rankings(criteria_matrices, alternative_matrices, criteria_names, alternative_names):
    """Calculate both baseline and level-adjusted rankings."""
    # First calculate baseline (treating all as strategic)
    baseline_priority_vectors = {}
    for name, (matrix, _) in criteria_matrices.items():
        # Calculate weights without level adjustment
        weights = calculate_fuzzy_synthetic_extent(matrix)

        # Store fuzzy weights
        baseline_priority_vectors[f"{name}_l"] = weights[:, 0]
        baseline_priority_vectors[f"{name}_m"] = weights[:, 1]
        baseline_priority_vectors[f"{name}_u"] = weights[:, 2]

        # Calculate and store defuzzified weights
        defuzz_weights = np.array([
            defuzzify_tfn(l, m, u) 
            for l, m, u in zip(weights[:, 0], weights[:, 1], weights[:, 2])
        ])
        baseline_priority_vectors[f"{name}_crisp"] = normalize_crisp_weights(defuzz_weights)

    # Then calculate level-adjusted results
    adjusted_priority_vectors = {}
    strategic_matrices = [
        matrix for (matrix, level) in criteria_matrices.values()
        if level == "Strategic"
    ]

    if not strategic_matrices:
        st.warning("No strategic-level respondents found. Using default weights.")
        strategic_avg = np.ones_like(next(iter(criteria_matrices.values()))[0])
    else:
        strategic_avg = np.mean(strategic_matrices, axis=0)

    # Calculate criteria weights
    for name, (matrix, level) in criteria_matrices.items():
        adjusted_matrix = adjust_weights_matrix(matrix, strategic_avg, level)
        weights = calculate_fuzzy_synthetic_extent(adjusted_matrix)

        # Store fuzzy weights
        adjusted_priority_vectors[f"{name}_l"] = weights[:, 0]
        adjusted_priority_vectors[f"{name}_m"] = weights[:, 1]
        adjusted_priority_vectors[f"{name}_u"] = weights[:, 2]

        # Calculate and store defuzzified weights
        defuzz_weights = np.array([
            defuzzify_tfn(l, m, u) 
            for l, m, u in zip(weights[:, 0], weights[:, 1], weights[:, 2])
        ])
        adjusted_priority_vectors[f"{name}_crisp"] = normalize_crisp_weights(defuzz_weights)

    # Calculate final criteria weights for both methods
    crisp_columns = [col for col in baseline_priority_vectors if col.endswith('_crisp')]
    baseline_criteria_weights = pd.DataFrame(baseline_priority_vectors)[crisp_columns].mean(axis=1).values
    adjusted_criteria_weights = pd.DataFrame(adjusted_priority_vectors)[crisp_columns].mean(axis=1).values

    # Only calculate alternative rankings if alternatives exist
    baseline_rankings_df = None
    adjusted_rankings_df = None

    if alternative_matrices and len(alternative_names) > 0:
        # Calculate baseline alternative rankings
        baseline_weights, baseline_indices = calculate_alternative_priorities(
            alternative_matrices,
            baseline_criteria_weights,
            adjust_levels=False
        )

        if len(baseline_weights) > 0:
            baseline_rankings_df = pd.DataFrame({
                'Alternative': [alternative_names[i] for i in baseline_indices],
                'Weight': baseline_weights,
                'Percentage': [f"{w*100:.2f}%" for w in baseline_weights]
            })

        # Calculate level-adjusted alternative rankings
        adjusted_weights, adjusted_indices = calculate_alternative_priorities(
            alternative_matrices,
            adjusted_criteria_weights,
            adjust_levels=True
        )

        if len(adjusted_weights) > 0:
            adjusted_rankings_df = pd.DataFrame({
                'Alternative': [alternative_names[i] for i in adjusted_indices],
                'Weight': adjusted_weights,
                'Percentage': [f"{w*100:.2f}%" for w in adjusted_weights]
            })

    return (baseline_rankings_df, baseline_priority_vectors, 
            adjusted_rankings_df, adjusted_priority_vectors)

def display_results(baseline_rankings_df, baseline_priority_vectors, 
                   adjusted_rankings_df, adjusted_priority_vectors, 
                   criteria_names):
    """Display both baseline and level-adjusted results."""

    # Calculate and display criteria weights for both methods
    st.header("Criteria Evaluation Results")

    # Baseline Results
    st.subheader("Baseline FAHP Results (No Level Adjustment)")
    crisp_columns = [col for col in baseline_priority_vectors if col.endswith('_crisp')]
    baseline_weights = pd.DataFrame(baseline_priority_vectors)[crisp_columns].mean(axis=1)
    baseline_weights.index = criteria_names

    baseline_criteria_df = pd.DataFrame({
        'Criterion': criteria_names,
        'Weight': baseline_weights,
        'Percentage': [f"{w*100:.2f}%" for w in baseline_weights]
    }).sort_values('Weight', ascending=False)

    st.write("Baseline Criteria Weights:")
    st.dataframe(
        baseline_criteria_df.style.format({
            'Weight': '{:.4f}'
        }).background_gradient(
            subset=['Weight'],
            cmap='YlOrRd'
        ),
        hide_index=True
    )

    # Level-adjusted Results
    st.subheader("Level-adjusted FAHP Results")
    adjusted_weights = pd.DataFrame(adjusted_priority_vectors)[crisp_columns].mean(axis=1)
    adjusted_weights.index = criteria_names

    adjusted_criteria_df = pd.DataFrame({
        'Criterion': criteria_names,
        'Weight': adjusted_weights,
        'Percentage': [f"{w*100:.2f}%" for w in adjusted_weights]
    }).sort_values('Weight', ascending=False)

    st.write("Level-adjusted Criteria Weights:")
    st.dataframe(
        adjusted_criteria_df.style.format({
            'Weight': '{:.4f}'
        }).background_gradient(
            subset=['Weight'],
            cmap='YlOrRd'
        ),
        hide_index=True
    )

    # Criteria Weight Visualizations
    st.subheader("Criteria Weight Visualizations")

    # Create four columns for the visualizations
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.write("Baseline Aggregated Weights")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        plt.bar(baseline_criteria_df['Criterion'], baseline_criteria_df['Weight'])
        plt.xticks(rotation=45)
        plt.title("Baseline Aggregated Weights")
        plt.tight_layout()
        st.pyplot(fig1)

    with col2:
        st.write("Level-adjusted Aggregated Weights")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        plt.bar(adjusted_criteria_df['Criterion'], adjusted_criteria_df['Weight'])
        plt.xticks(rotation=45)
        plt.title("Level-adjusted Aggregated Weights")
        plt.tight_layout()
        st.pyplot(fig2)

    with col3:
        st.write("Baseline Individual Weights")
        fig3, ax3 = plt.subplots(figsize=(10, 6))

        # Get individual respondent weights for baseline
        respondents = set(col.split('_')[0] for col in baseline_priority_vectors if col.endswith('_crisp'))
        df_baseline_respondents = pd.DataFrame()

        for resp in respondents:
            df_baseline_respondents[resp] = baseline_priority_vectors[f"{resp}_crisp"]

        df_baseline_respondents.index = criteria_names
        df_baseline_respondents.plot(kind='bar', ax=ax3)
        plt.title("Baseline Individual Weights")
        plt.xticks(rotation=45)
        plt.legend(title="Respondents", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        st.pyplot(fig3)

    with col4:
        st.write("Level-adjusted Individual Weights")
        fig4, ax4 = plt.subplots(figsize=(10, 6))

        # Get individual respondent weights for level-adjusted
        df_adjusted_respondents = pd.DataFrame()

        for resp in respondents:
            df_adjusted_respondents[resp] = adjusted_priority_vectors[f"{resp}_crisp"]

        df_adjusted_respondents.index = criteria_names
        df_adjusted_respondents.plot(kind='bar', ax=ax4)
        plt.title("Level-adjusted Individual Weights")
        plt.xticks(rotation=45)
        plt.legend(title="Respondents", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        st.pyplot(fig4)

    # Alternative Rankings
    if baseline_rankings_df is not None and adjusted_rankings_df is not None:
        st.header("Alternative Evaluation Results")

        # Baseline Alternative Rankings
        st.subheader("Baseline Alternative Rankings")
        st.dataframe(
            baseline_rankings_df.style.format({
                'Weight': '{:.4f}'
            }).background_gradient(
                subset=['Weight'],
                cmap='YlOrRd'
            ),
            hide_index=True
        )

        # Level-adjusted Alternative Rankings
        st.subheader("Level-adjusted Alternative Rankings")
        st.dataframe(
            adjusted_rankings_df.style.format({
                'Weight': '{:.4f}'
            }).background_gradient(
                subset=['Weight'],
                cmap='YlOrRd'
            ),
            hide_index=True
        )

        # Alternative Visualizations
        st.subheader("Alternative Ranking Visualizations")
        col5, col6 = st.columns(2)
        col7, col8 = st.columns(2)

        with col5:
            st.write("Baseline Aggregated Rankings")
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            plt.bar(baseline_rankings_df['Alternative'], baseline_rankings_df['Weight'])
            plt.xticks(rotation=45)
            plt.title("Baseline Aggregated Rankings")
            plt.tight_layout()
            st.pyplot(fig5)

        with col6:
            st.write("Level-adjusted Aggregated Rankings")
            fig6, ax6 = plt.subplots(figsize=(10, 6))
            plt.bar(adjusted_rankings_df['Alternative'], adjusted_rankings_df['Weight'])
            plt.xticks(rotation=45)
            plt.title("Level-adjusted Aggregated Rankings")
            plt.tight_layout()
            st.pyplot(fig6)

        with col7:
            st.write("Baseline Individual Rankings")
            fig7, ax7 = plt.subplots(figsize=(10, 6))

            # Plot individual criterion contributions for baseline
            baseline_criterion_rankings = {}
            for criterion in criteria_names:
                baseline_criterion_rankings[criterion] = baseline_rankings_df['Weight'] * baseline_weights[criterion]

            df_baseline_criterion = pd.DataFrame(baseline_criterion_rankings)
            df_baseline_criterion.index = baseline_rankings_df['Alternative']
            df_baseline_criterion.plot(kind='bar', ax=ax7, stacked=True)
            plt.title("Baseline Rankings by Criterion")
            plt.xticks(rotation=45)
            plt.legend(title="Criteria", bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
            st.pyplot(fig7)

        with col8:
            st.write("Level-adjusted Individual Rankings")
            fig8, ax8 = plt.subplots(figsize=(10, 6))

            # Plot individual criterion contributions for level-adjusted
            adjusted_criterion_rankings = {}
            for criterion in criteria_names:
                adjusted_criterion_rankings[criterion] = adjusted_rankings_df['Weight'] * adjusted_weights[criterion]

            df_adjusted_criterion = pd.DataFrame(adjusted_criterion_rankings)
            df_adjusted_criterion.index = adjusted_rankings_df['Alternative']
            df_adjusted_criterion.plot(kind='bar', ax=ax8, stacked=True)
            plt.title("Level-adjusted Rankings by Criterion")
            plt.xticks(rotation=45)
            plt.legend(title="Criteria", bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
            st.pyplot(fig8)

    # Export options
    st.subheader("Export Results")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Export criteria weights
        baseline_criteria_df.to_excel(writer, sheet_name='Baseline Criteria Weights', index=False)
        adjusted_criteria_df.to_excel(writer, sheet_name='Adjusted Criteria Weights', index=False)

        # Export full details
        pd.DataFrame(baseline_priority_vectors).to_excel(writer, sheet_name='Baseline Details', index=True)
        pd.DataFrame(adjusted_priority_vectors).to_excel(writer, sheet_name='Adjusted Details', index=True)

        # Export alternative rankings if available
        if baseline_rankings_df is not None:
            baseline_rankings_df.to_excel(writer, sheet_name='Baseline Alternative Rankings', index=False)
            adjusted_rankings_df.to_excel(writer, sheet_name='Adjusted Alternative Rankings', index=False)

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

    num_criteria, num_alternatives, num_respondents, criteria_names, alternative_names = input_section()
    respondents = collect_respondent_data(num_respondents)

    if not all(name for name, _ in respondents):
        st.error("Please enter all respondent names")
        return

    criteria_matrices, alternative_matrices = collect_comparisons(
        num_criteria,
        num_alternatives,
        criteria_names,
        alternative_names,
        respondents
    )

    if criteria_matrices is not None and alternative_matrices is not None:
        if st.button("Calculate"):
            with st.spinner("Calculating results..."):
                baseline_rankings_df, baseline_priority_vectors, adjusted_rankings_df, adjusted_priority_vectors = calculate_final_rankings(
                    criteria_matrices,
                    alternative_matrices,
                    criteria_names,
                    alternative_names
                )
                display_results(
                    baseline_rankings_df,
                    baseline_priority_vectors,
                    adjusted_rankings_df,
                    adjusted_priority_vectors,
                    criteria_names
                )

if __name__ == "__main__":
    main()
