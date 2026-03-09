# Part 3: Data Cleaning & Feature Engineering

Goal: Get the data ready for the machine. This is the most crucial hands-on part.

Data Cleaning (10 mins):

Handle Missing Data: What do we do if a listing has a blank "cleaning fee"? Drop it or fill it?

Convert Strings to Numbers: Machines only understand math. Show how to convert a neighborhood name like "Brooklyn" into numerical categories (One-Hot Encoding) or converting a "$150" string into the integer 150.

Visualizing the Data (10 mins):

Use Matplotlib or Seaborn to plot a quick chart (e.g., a scatter plot of Price vs. Number of Bedrooms).

Explain how visualization helps uncover hidden correlations and outliers (e.g., a 1-bedroom listed for $10,000/night).

Feature Selection & Splitting (10 mins):

Feature Extraction/Selection: Decide which columns we actually need. Drop irrelevant columns like "Host Name" or "Image URL."

Split the Dataset: Explain why we cannot test the model on the data it learned from. Show the code to split the data into Training (80%) and Testing (20%).