import importlib.util
import pprint
import re
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO

from pytest_examples import CodeExample
from pytest_examples.config import ExamplesConfig
from pytest_examples.run_code import InsertPrintStatements
from pytest_examples.traceback import create_example_traceback


@dataclass
class RunRequest:
    tag: str
    code: str


VALID_EXAMPLE = """\
Please run this example on main:
```py
try:
    assert False
except Exception as e:
    print(e)

print(123)
```
"""


DEFAULT_TAG = 'main'


def parse_comment(comment: str) -> RunRequest:
    match = re.match(r'(please run this example)(?: on ([^\n:\s]+)){0,1}', comment, flags=re.IGNORECASE)
    if not match or not match.group(1):
        raise ValueError(
            f'Could not parse comment. Expected a comment of the format:\n{VALID_EXAMPLE}'
        )
    tag = match.group(2)
    if not tag:
        tag = DEFAULT_TAG
        end = match.end(1)
    else:
        end = match.end(2)
    comment = comment[end:]
    match = re.match(r'[\s\S]*```[^\n]*\n(.*?)(?=\n```)[\s\S]*', comment)
    if not match:
        raise ValueError(
            f'Could not parse comment. Expected a comment of the format:\n{VALID_EXAMPLE}'
        )
    code: str = match.group(1)
    return RunRequest(tag=tag, code=code)


def run_example(code: str) -> str:
    config = ExamplesConfig()

    with NamedTemporaryFile(suffix='.py') as file:
        with open(file.name, mode='w') as f:
            f.write(code)

        python_file = Path(file.name)

        example = CodeExample.create(source=code)

        spec = importlib.util.spec_from_file_location('__main__', str(python_file))
        assert spec is not None
        module = importlib.util.module_from_spec(spec)

        insert_print = InsertPrintStatements(python_file, config, enable=True)

        loader = spec.loader
        assert loader is not None

        try:
            with insert_print:
                loader.exec_module(module)
        except KeyboardInterrupt:
            print('KeyboardInterrupt in example')
        except Exception as exc:
            example_tb = create_example_traceback(exc, str(python_file), example)
            if example_tb:
                raise exc.with_traceback(example_tb)
            else:
                raise exc

        new_code = insert_print.updated_print_statements(example)
        assert new_code is not None
        return new_code



def main_eval_example(stdin: IO[str] = sys.stdin, stdout: IO[str] = sys.stdout) -> None:
    """Run the code block in the comment"""
    request = parse_comment(stdin.read())
    try:
        res = run_example(request.code)
        stdout.write(res)
        stdout.write('\n')
    except Exception as e:
        stdout.write('Error evaluating example. Please remember to **catch and print all exceptions**. Output:\n')
        # put the error in a code block
        stdout.write('```')
        pprint.pprint(e, stream=stdout)
        stdout.write('```')



def main_get_tags(stdin: IO[str] = sys.stdin, stdout: IO[str] = sys.stdout) -> None:
    """Parse the tag to check out from the comment"""
    try:
        request = parse_comment(stdin.read())
        stdout.write(request.tag)
    except Exception as e:
        stdout.write('Error parsing request. Maybe you have a malformed comment? Expected a comment matching the regex ')
        # put the error in a code block
        stdout.write('```')
        pprint.pprint(e, stream=stdout)
        stdout.write('```')


if __name__ == '__main__':
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    get_tags_parser = subparsers.add_parser('get-tag')
    evaluate = subparsers.add_parser('evaluate')
    args = parser.parse_args()
    command: str = args.command
    if command == 'get-tag':
        main_get_tags()
    elif command == 'evaluate':
        main_eval_example()
    else:
        raise ValueError(f"Unknown command '{command}'")
