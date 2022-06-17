
import sys

from runners import MainRunner

from .experiment_configurator import ExperimentConfigurator

## use this in most cases
modules_creator = ExperimentConfigurator()
runner = MainRunner(sys.argv[1], modules_creator)
runner.run_loop()
