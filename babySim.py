import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import poisson
from fractions import Fraction


# Set the page configuration (optional)
st.set_page_config(
    page_title="Baby Gender Simulation",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("👶 Baby Gender Simulation")

# Define the slider options
options_lambda = [0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 7, 10, 15, 20, 30, 50, 100]
options_days = [1, 2, 4, 6, 10, 15, 20, 40, 70, 100, 300, 1000, 10000]

# Create four columns: three for sliders and one for the button
slider_col1, slider_col2, slider_col3, button_col = st.columns(4)

with slider_col1:
    lambda_ = st.select_slider(
        "Avg # Babies Per Day",
        options=options_lambda,
        value=4.0
    )

with slider_col2:
    p = st.slider(
        "Probability Baby is Boy",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        format="%.2f"
    )

with slider_col3:
    n_days = st.select_slider(
        "Number of Days Observed",
        options=options_days,
        value=300
    )

with button_col:
    # Place the "Run Simulation" button in the fourth column
    rerun_simulation = st.button("🔄 Rerun Simulation")

st.markdown("---")

# Define bins and labels for gender ratio distribution at the global scope
bins = [-0.1, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Include -0.1 to capture 0% correctly
labels = ['0', '.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9', '1']

# Function to perform the simulation
def run_baby_gender_simulation(lambda_, p, n_days):
    # Simulate number of babies born each day using Poisson distribution
    n_babies_per_day = np.random.poisson(lambda_, n_days)
    
    
    percent_males_per_day = []
    total_boys = 0
    total_girls = 0

    for n_babies in n_babies_per_day:
        if n_babies == 0:
            continue  # Skip days with no babies
        # Process days with babies
        genders = np.random.binomial(1, p, n_babies)  # 1: Boy, 0: Girl
        num_boys = np.sum(genders)
        num_girls = n_babies - num_boys
        percent_males = (num_boys / n_babies) * 100
        percent_males_per_day.append(percent_males)
        total_boys += num_boys
        total_girls += num_girls
    
    # Convert list to NumPy array for processing
    percent_males_array = np.array(percent_males_per_day)
    
    # Exclude NaN values (days with no babies)
    valid_percentages = percent_males_array[~np.isnan(percent_males_array)]
    
    # Assign each day's percentage to a bin
    bin_indices = np.digitize(valid_percentages, bins)
    
    # Count the number of days in each bin
    df = pd.DataFrame({'percent_males': percent_males_per_day})
    df['bins'] = pd.cut(df['percent_males'], bins=bins, labels=labels, include_lowest=True)
    gender_ratio_counts = df['bins'].value_counts().sort_index().to_dict()
    
    # Prepare data for the Poisson distribution plot
    if len(n_babies_per_day) > 0:
        max_babies = np.max(n_babies_per_day)
    else:
        max_babies = 0
    x_values = np.arange(0, max_babies + 1)
    simulation_counts = np.bincount(n_babies_per_day, minlength=len(x_values))
    theoretical_probs = poisson.pmf(x_values, lambda_)
    theoretical_counts = theoretical_probs * n_days
    
    # Prepare the results dictionary
    results = {
        'n_babies_per_day': n_babies_per_day,
        'percent_males_per_day': percent_males_per_day,
        'gender_ratio_counts': gender_ratio_counts,
        'total_boys': total_boys,
        'total_girls': total_girls,
        'x_values': x_values,
        'simulation_counts': simulation_counts,
        'theoretical_counts': theoretical_counts
    }
    
    return results

# Run the simulation whenever parameters change or the button is pressed
simulation_results = run_baby_gender_simulation(lambda_, p, n_days)

# Extract results
gender_ratio_counts = simulation_results['gender_ratio_counts']
total_boys = simulation_results['total_boys']
total_girls = simulation_results['total_girls']
x_values = simulation_results['x_values']
simulation_counts = simulation_results['simulation_counts']
theoretical_counts = simulation_results['theoretical_counts']


# Create two columns for the plots
plot_col1, plot_col2 = st.columns(2)

# Plot the "# of Days with X Number of Babies Born"
with plot_col1:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    width = 0.35  # the width of the bars
    ax1.bar(x_values - width/2, simulation_counts, width=width, label='Simulation', color='skyblue')
    ax1.bar(x_values + width/2, theoretical_counts, width=width, label='Theoretical', color='orange')
    ax1.set_xlabel('Number of Babies Born')
    ax1.set_ylabel('Number of Days')
    ax1.set_title('Distribution of Birth Counts per Day')
    ax1.legend()
    
    # Set x-axis ticks to show only integer values
    ax1.set_xticks(x_values)
    ax1.set_xticklabels(x_values)
    
    # Optional: Adjust x-axis limits
    ax1.set_xlim([x_values.min() - 1, x_values.max() + 1])
    
    # Remove grid lines
    ax1.grid(False)
    # Remove black framing around the plot
    for spine in ax1.spines.values():
        spine.set_visible(False)
    
    st.pyplot(fig1)

with plot_col2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    counts = [gender_ratio_counts.get(label, 0) for label in labels]
    positions = np.arange(len(labels))
    bars = ax2.bar(positions, counts, color='purple')
    ax2.set_xlabel('Percentage of Babies that Day are Male')
    ax2.set_ylabel('Number of Days')
    ax2.set_title('Distribution of daily Male-Female Percentages')
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels, rotation=45)

    # Remove grid lines and spines
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # Add annotations
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{int(height)}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    st.pyplot(fig2)
# Display Simulation Results
st.markdown("---")
st.subheader("📊 Simulation Results")
cols = st.columns(2)

with cols[0]:
    st.metric("Total Boys Born", int(total_boys))
    

with cols[1]:
    st.metric("Total Girls Born", int(total_girls))
    
