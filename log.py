import sys

from termcolor import colored

# Basic logging infrastructure. Simpler than the python 'logging' module.
def info(message):
    print(colored(message, 'blue'), file=sys.stderr)
def warn(message):
    print(colored(message, 'yellow'), file=sys.stderr)
def error(message):
    print(colored(message, 'red'), file=sys.stderr)
