# Contributing to Test-of-Time
:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

Welcome to the Test-of-time community and thank you for your contribution to its continued legacy. 
We compiled this page with practical instructions and further resources to help you get started.

Please come join us on the NeuralProphet (our main framework) [Slack](https://join.slack.com/t/neuralprophet/shared_invite/zt-sgme2rw3-3dCH3YJ_wgg01IXHoYaeCg), you can message any core dev there.

## Get Started On This
We have a GitHub section tracking all [issues](https://github.com/ourownstory/test-of-time/issues) to be solved. 
There are some issues marked with the label 'good first issue'. These issues are ideal to get started on for you as a beginner.
Feel free to assign yourself to one on GitHub and have a try solving it. In case you need help, don't hestitate to reach out
via the commenting function in GutHub or Slack. 

## Process
Here's a great [beginner's guide to contributing to a GitHub project](https://akrabat.com/the-beginners-guide-to-contributing-to-a-github-project/#to-sum-up). 

In Summary: 
* Fork the project & clone locally.
* Create an upstream remote and sync your local copy before you branch.
* Branch for each separate piece of work.
* Do the work, write good commit messages, and read the CONTRIBUTING file if there is one.
* Push to your origin repository.
* Create a new PR in GitHub. It is helpful to open a PR right when you start on a new issue to show that work is in progress.
* Respond to any code review feedback.

Please make sure to include tests and documentation with your code.

## Dev Install
Before starting it's a good idea to first create and activate a new virtual environment:
```
python3 -m venv <path-to-new-env>
source <path-to-new-env>/bin/activate
```
Now you can install test-of-time:

```
git clone <copied link from github>
cd test-of-time
pip install -e ".[dev]"
```

Please don't forget to run the dev setup script to install the hooks for black and pytest, and set git to fast forward only:
```
tot_dev_setup.py
git config pull.ff only 
```

Notes: 
* Including the optional `-e` flag will install test-of-time in "editable" mode, meaning that instead of copying the files into your virtual environment, a symlink will be created to the files where they are.
* The `tot_dev_setup` command runs the dev-setup script which installs appropriate git hooks for Black (pre-commit) and PyTest (pre-push).
* setting git to fast-forward only prevents accidental merges when using `git pull`.
* To run tests without pushing (or when the hook installation fails), run from test-of-time folder: `pytest -v`
* To run black without commiting (or when the hook installation fails): `python3 -m black {source_file_or_directory}` 
* If running `tot_dev_setup.py` gives you a `no such file` error, try running `python ./scripts/tot_dev_setup.py`

In case you spot an error in this description, please reach out.

## Docstring

Docstrings need to be formatted according to [NumPy Style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy). 
Please refer to [Pandas Docstring Guide](https://pandas.pydata.org/pandas-docs/stable/development/contributing_docstring.html#) for best practices.

The length of line inside docstrings block must be limited to 80 characters to fit into Jupyter documentation popups.

You can check for adherence to the style guide by running:
```sh
pydocstyle --convention=numpy path/my_file.py
```
(You may need to install the tool first. On Linux: `sudo apt install pydocstyle`.)


#### Example 
See how Pandas does this for `melt` in their [melt documentation page](https://pandas.pydata.org/docs/reference/api/pandas.melt.html) and how it looks in the [melt docstring](https://github.com/pandas-dev/pandas/blob/v1.4.1/pandas/core/shared_docs.py#L153).

Docstring architecture sample:

```
def return_first_elements(n=5):
    """
    Return the first elements of a given Series.

    This function is mainly useful to preview the values of the
    Series without displaying all of it.

    Parameters
    ----------
    n : int
        Number of values to return.

    Return
    ------
    pandas.Series
        Subset of the original series with the n first values.

    See Also
    --------
    tail : Return the last n elements of the Series.
    Examples
    --------
    If you have multi-index columns:
    >>> df.columns = [list('ABC'), list('DEF')]
    >>> df
       A  B  C
       D  E  F
    0  a  1  2
    1  b  3  4
    2  c  5  6
    """
    return self.iloc[:n]
```

## Typing

We try to use type annotations across the project to improve code readability and maintainability.

Please follow the official python recommendations for [type hints](https://docs.python.org/3/library/typing.html) and [PEP-484](https://peps.python.org/pep-0484/).

### Postponing the evaluation type annotations and python version

The Postponed Evaluation of Annotations [PEP 563](https://docs.python.org/3/whatsnew/3.7.html#pep-563-postponed-evaluation-of-annotations) provides major benefits for type annotations. To use them with our currently support python versions we must use the following syntax:

```python
from __future__ import annotations
```

### Circular imports with type annotations

When using type annotations, you may encounter circular imports. To avoid this, you can use the following pattern based on the [typing.TYPE_CHECKING](https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING) constant:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

# Imports only needed for type checking
if TYPE_CHECKING:
    from my_module import MyType
```

## Testing and Code Coverage

We are using `PyTest` to run tests within our projects. All tests can be found in `tests/` directory. 

All tests can be triggered via the command: 

```bash
pytest tests -v
```

Running specific tests can be done by running the command: 

```bash
pytest tests -k "name_of_test"
```

We are using [pytest-cov](https://pypi.org/project/pytest-cov/) and [codecov](https://app.codecov.io/gh/ourownstory/neural_prophet) to create transparent code coverage reports.
To locally trigger and output a code coverage report via the commandline, run the following command: 

```bash
pytest tests -v --cov=./
```


## Continous Integration

We are using Github Actions to setup a CI pipeline. The creation as well as single commits to a pull request trigger the CI pipeline.

Currently there is one workflow called `.github/worklfows/ci.yml` to trigger testing, create code coverage reports via [pytest-cov](https://pypi.org/project/pytest-cov/) and subsequently uploading reports via [codecov](https://app.codecov.io/gh/ourownstory/neural_prophet) for the major OS systems (Linux, Mac, Windows). 


## Style
We deploy Black, the uncompromising code formatter, so there is no need to worry about style. Beyond that, where reasonable, for example for docstrings, we follow the [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html)

As for Git practices, please follow the steps described at [Swiss Cheese](https://github.com/ourownstory/swiss-cheese/blob/master/git_best_practices.md) for how to git-rebase-squash when working on a forked repo. (Update: all PR are now squashed, so you can skip this step, but it's still good to know.)

### String formatting
Please use the more readable [f-string formatting style](https://docs.python.org/3/tutorial/inputoutput.html).

## Pull requests
Add one or multiple **labels** to your Pull request to be able to grasp its purpose at one glance.
List of current labels:

* documentation
* enhancement
* model
* plotting
* tutorial
* workflow
* help wanted
* need fix
* need review

  
## Issues
In case you spot a bug in the code or would like to suggest an improvement point, 
it is best practice to open a new issue to address it. Add one or multiple **labels** to your newly created issue. 
* good first issue
* bug
* duplicate
* wontfix
* plotting
* tutorial
* workflow
* model
* documentation




