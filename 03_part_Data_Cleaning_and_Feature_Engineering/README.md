# Part 3: Data Cleaning & Feature Engineering

Machine learning algorithms are essentially giant math equations; they cannot understand text, they hate blank spaces, and they get confused by irrelevant information. To build a reliable model to predict Airbnb prices in Vienna, we need to transform our raw, messy data into a pristine, mathematically digestible format.

## Data Cleaning

Before we can teach a model anything, we have to clean up the dataset. Normalizing data from diverse sources and fixing errors is a multi-step process. Here are the most common steps we take:
- **Removing Duplicates**: Sometimes, the same listing gets scraped or recorded twice. We must identify and drop these identical rows so they don't unfairly skew our model's understanding.
- **Handling Missing Values (NaNs)**: Real data is full of holes. If a host didn't fill the cells like "number of bathrooms," those cells show up as NaN (Not a Number). We must decide whether to completely drop the rows missing this data, or fill them in (impute) with a logical guess, like the average value.
- **Cleaning Strings**: Text data often contains symbols that break mathematical models. For example, a price written as "€150.00" needs to have the "€" symbol and the commas stripped away so we are left with pure data.
- **Converting Values to Numbers**: Algorithms only speak math. Any data that remains as a text string must be converted into an integer or a decimal.
- **One-Hot Encoding**: What do we do with categorical text, like a neighborhood name (e.g., "Innere Stadt" or "Leopoldstadt")? We use a technique called One-Hot Encoding. This converts categorical categories into a series of binary columns (0s and 1s) representing whether a listing belongs to that neighborhood or not, allowing the algorithm to mathematically process the location.

## Visualizing the Data

Before we finalize our data, it is crucial to actually *look* at it. An important aspect of the data scientist's toolkit is the power to visualize data using excellent Python libraries such as **MatPlotLib** or **Seaborn**.

Why is this important? Because looking at a spreadsheet with 10,000 rows tells you almost nothing. Representing your data visually allows you to:
-  **Uncover Hidden Correlations:** A scatter plot might instantly reveal a strong relationship between the number of bedrooms and the nightly price, or show how prices drop as the distance from the city center increases. We can leverage these patterns later.
- **Spot Outliers and Unbalanced Data:** Visualizations make anomalies painfully obvious. If someone accidentally listed a 1-bedroom apartment for €10,000 a night, a chart will immediately flag this outlier so we can remove it before it confuses our model.

## Defining Our Variables: Features and Target

As we refine our data, we need to clearly define what we are inputting and what we want to get out.
- **The Target ($y$):** This is the ultimate thing you are trying to predict. In our case, the target represents the answer to the question we are asking: *What is the nightly price of the Airbnb?*
- **The Features ($X$):** A feature is a measurable property of your data. These are the input variables used to train the model, usually expressed as column headings like 'bedrooms', 'bathrooms', or 'has_wifi'.

## Feature Selection & Feature Extraction

How do you know which variables to choose when building a model? Throwing every single column at the algorithm is usually a bad idea. We go through a deliberate process to choose the right variables for the most performant model:
- **Feature Selection:** This returns a targeted subset of the features. We actively decide which columns we actually need and drop the rest. For instance, the "Host's First Name" or the "Image URL" will not help us predict the price of an apartment, so we delete those columns entirely to reduce noise.
- **Feature Extraction:** This creates *new* features from functions of the original features. For example, if our dataset has a "Host Joined Date" column, the raw date isn't very helpful. But we could extract a new, highly useful numerical feature from it: "Years as Host."

## Splitting Your Dataset

Finally, prior to training, you must split your pristine dataset into two parts of unequal size.
- **Training Set (Usually 80%):** This part of the dataset is fit to your model to train it. It constitutes the majority of the data and acts as the "textbook" the algorithm studies to learn the patterns between the features and the target.
- **Testing Set (Usually 20%):** This is an independent, unseen group of data. We hide this from the model during the training phase. Once the model thinks it has learned the rules, we use this test set to give it a "final exam" to confirm its true performance. If we tested the model on the exact same data it trained on, it would be like giving a student the answer key before the test!
