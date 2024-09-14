import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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
options_lambda = [0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100, 300, 1000]
options_days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 12, 14, 16, 18, 20,
               25, 30, 35, 40, 50, 70, 100, 300, 1000]

# Create three columns for sliders
slider_col1, slider_col2, slider_col3 = st.columns(3)

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
        value=10
    )

# Add a "Run Simulation" button
run_simulation = st.button("🔄 Run Simulation")

# Define bins and labels for gender ratio distribution at the global scope
bins = [0, 20, 40, 60, 80, 100]  # Include -0.1 to capture 0% correctly
labels = ['0%', '20%', '40%', '60%', '80%', '100%']

# Function to perform the simulation
def run_baby_gender_simulation(lambda_, p, n_days):
    # Simulate number of babies born each day using Poisson distribution
    n_babies_per_day = np.random.poisson(lambda_, n_days)
    
    # Simulate genders for each baby born each day
    percent_males_per_day = []
    total_boys = 0
    total_girls = 0
    
    for n_babies in n_babies_per_day:
        if n_babies > 0:
            genders = np.random.binomial(1, p, n_babies)  # 1: Boy, 0: Girl
            num_boys = np.sum(genders)
            num_girls = n_babies - num_boys
            percent_males = (num_boys / n_babies) * 100
            percent_males_per_day.append(percent_males)
            total_boys += num_boys
            total_girls += num_girls
        else:
            percent_males_per_day.append(np.nan)  # No babies born on this day
    
    # Convert list to NumPy array for processing
    percent_males_array = np.array(percent_males_per_day)
    
    # Exclude NaN values (days with no babies)
    valid_percentages = percent_males_array[~np.isnan(percent_males_array)]
    
    # Assign each day's percentage to a bin
    bin_indices = np.digitize(valid_percentages, bins)
    
    # Count the number of days in each bin
    gender_ratio_counts = {label: 0 for label in labels}
    for idx, label in enumerate(labels):
        count = np.sum(bin_indices == idx + 1)
        gender_ratio_counts[label] = count
    
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
    ax1.set_title('# of Days with X Number of Babies Born')
    ax1.legend()
    
    # Remove grid lines
    ax1.grid(False)
    # Remove black framing around the plot
    for spine in ax1.spines.values():
        spine.set_visible(False)
    
    st.pyplot(fig1)

# Plot the Distribution of Gender Ratios
with plot_col2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    # Use the labels defined globally
    ratio_labels = labels  # ['0%', '20%', '40%', '60%', '80%', '100%']
    counts = [gender_ratio_counts[label] for label in labels]
    bars = ax2.bar(ratio_labels, counts, color='purple')
    ax2.set_xlabel('Percentage of Male Babies')
    ax2.set_ylabel('Number of Days')
    ax2.set_title('# Days that had certain % of males')
    
    # Remove grid lines
    ax2.grid(False)
    # Remove black framing around the plot
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    # Add annotations to each bar
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    st.pyplot(fig2)

# Display Simulation Results
st.markdown("---")
st.subheader("📊 Simulation Results")
cols = st.columns(2)

with cols[0]:
    st.metric("Total Days Observed", n_days)
    st.metric("Total Boys Born", int(total_boys))

with cols[1]:
    st.metric("Total Girls Born", int(total_girls))
    # Calculate and display the overall boy-to-girl ratio
    if total_girls > 0:
        # Convert to a fraction and limit the denominator to a reasonable size
        ratio_fraction = Fraction(total_boys, total_girls).limit_denominator()
        st.metric("Boys-to-Girls Ratio", f"{ratio_fraction.numerator}:{ratio_fraction.denominator}")
    elif total_boys > 0:
        st.metric("Boys-to-Girls Ratio", "∞:1")
    else:
        st.metric("Boys-to-Girls Ratio", "Undefined")

# Re-run the simulation when the button is pressed
if run_simulation:
    st.experimental_rerun()
