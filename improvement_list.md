### file-independent
* <del>add pyright<del>
* <del>add codecov<del>
* investigate computing time reduction potential, e.g. via multiprocessing, slicing, or .apply
* vectorize all relevant function

### dataset.py
* implement an easy .load() method

### df_utils.py
* <del>implement split_df() function, where we split the df dependend on n_lags and doesn't become visible to the user<del>
* Review if all functions intended private are private. 

### utils.py
* check if all private intended functions are private
* Rename and if needed or further split util related .py files. Convention files containing class name with noun, files containing no class name with verb. 

### experiment.py
* adjust the experiment name or params description to be less extensive
* transforming forecast_df to a class to introduce user friendly class methods?
* rethink the input mapping (data and model params) for model initialization

### benchmark.py
* restructure benchmark output as structured dict

### models.py
* <del>rethink class methods of LinearRegression model<del>
* <del>Docstring for all model class methods<del>
* Switch “ds” to “time”
* delete docstring of imported models 
* Rethink if we want the predicted df to be added up with Nans for all models (as it is now) or simplify and not add them
