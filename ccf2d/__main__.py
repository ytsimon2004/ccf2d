from argclz.commands import parse_command_args

from .main_init import InitOptions
from .main_register import RegisterOptions
from .main_view import ViewOptions

def main():
    parse_command_args(
        usage='python -m ccf2d ...',
        description='ccf2d cli usage',
        parsers=dict(
            init=InitOptions,
            register=RegisterOptions,
            view=ViewOptions
        )
    )
