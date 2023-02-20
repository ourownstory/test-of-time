### file-independent
* add pyright
* add codecov
* investigate omputing time reduction potential, e.g. via multiprocessing, slicing, or .apply

### dataset.py
* implement an easy .load() method

### df_utils.py
* implement split_df() function, where we split the df dependend on n_lags and doesn't become visible to the user
* Review if all functions intended private are private. 

### utils.py
* check if all private intended functions are private
* Rename and if needed or further split util related .py files. Convention files containing class name with noun, files containing no class name with verb. 

### experiment.py
* adjust the eperiment name or params description to be less extensive
* tranforming forecast_df to a class to introduce user friendly class methods
* rethink the input mapping (data and model params) for model initialization

### benchmark.py
* restructure benchmark output as structured dict

### models.py
* rethink clas methods of LinearRegression model
* Docstring for all model class methods
* Switch “ds” to “time”
* delete docstring of imported models 
