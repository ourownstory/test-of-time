[![Pypi_Version](https://img.shields.io/pypi/v/test-of-time.svg)](https://pypi.org/project/test-of-time/)
[![Python Version](https://img.shields.io/badge/python-3.7+-blue?logo=python)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/ourownstory/test-of-time/actions/workflows/ci.yml/badge.svg)](https://github.com/ourownstory/test-of-time/actions/workflows/ci.yml)
[![Slack](https://img.shields.io/badge/slack-@neuralprophet-CF0E5B.svg?logo=slack&logoColor=white&labelColor=3F0E40)](https://neuralprophet.slack.com/join/shared_invite/zt-sgme2rw3-3dCH3YJ_wgg01IXHoYaeCg#/shared-invite/email)
[![Downloads](https://static.pepy.tech/personalized-badge/test-of-time?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/test-of-time)


# Test-of-time
**Test-of-time** is a small framework to **benchmark time-series forecasting models** via backtesting. It offers a simple 
template for benchmarking as well as a variety of ready-to-use datasets and models. 

## Vision
Our vision is to become the central hub for benchmarking time series forecasts. 
As a user, you need to either (1) find a model that is best suited for your use case, or (2) evaluate the performance of your model 
for different use cases and derive its improvement potential.

We enable you to do this by integrating commonly used datasets and available models from open source or existing 
frameworks. 
We bundle these resources in one place and make them easily usable through a lean benchmarking template.
This enables fast benchmarking as a standard. In addition, we provide you with flexibility though customizable features, 
such as customizing the individual experiments in a benchmark, adding your own models or dataset.

## Install
You can now install test-of-time directly with pip:
```shell
pip install test-of-time
```

### Install options
If you would like the most up to date version, you can instead install direclty from github:
```shell
git clone <copied link from github>
cd test-of-time
pip install .
```

##Features

### Core framework features
* offers a simple benchmarking template to compare multiple models on multiple datasets via backtesting
* provides various error metrics
* provides a selection of ready-to-use models and datasets

### Add-on features
* offers to add your own model

For a list of past changes, please refer to the [releases page](https://github.com/ourownstory/test-of-time/releases).


## Tutorials & Examples 
You can find a first tutorial on how to use the framework in the [tutorial section](tutorials/BenchmarkingTemplates.ipynb).

## Contribution
We compiled a [Contributing to Test-of-Time](CONTRIBUTING.md) page with practical instructions and further resources 
to help you become part of the family.

## Community
#### Discussion and Help
If you have any question or suggestion, you can participate with [our community right here on Github](https://github.com/ourownstory/test-of-time/discussions) (coming soon)

#### Slack Chat
We also have an active [Slack community](https://join.slack.com/t/neuralprophet/shared_invite/zt-sgme2rw3-3dCH3YJ_wgg01IXHoYaeCg) for our main framework NeuralProphet.
Come and join the conversation!

## About
Test-of-time is a sub-framework of the open-source community project NeuralProphet. It is supported by awesome 
people like you. If you are interested in joining the project, please feel free to reach out to Oskar - 
email on the [NeuralProphet Paper](https://arxiv.org/abs/2111.15397).