# Part 4: Training, Evaluation, and Model Usage

Goal: Build the model, see how wrong it is, and make a real prediction.

Decide & Train (10 mins):

Select simple regression models. Start with Linear Regression (the simplest) and maybe introduce a Decision Tree Regressor to show how different algorithms perform.

Show the magic line of code: model.fit(X_train, y_train). Explain that this is where the math happens and the model learns the relationship between the features and the price.

Evaluate the Model (10 mins):

Use the testing data to gauge performance: model.predict(X_test).

Compare the predicted prices against the actual prices. Calculate a simple error metric (like Mean Absolute Error: "On average, our model is off by $20").

Overfitting vs. Underfitting: Explain these concepts. Is our model memorizing the training data (overfit) or failing to learn anything useful (underfit)?

Parameter Tuning & Prediction (10 mins):Briefly mention that we can tweak 'hyperparameters' to make the model smarter.The Grand Finale: Create a brand new, fake Airbnb listing right there in the class (e.g., "Let's list a 2-bedroom in downtown with a pool"). Feed this new $X$ into the model and reveal the predicted price $y$.