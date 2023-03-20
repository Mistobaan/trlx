# flake8: noqa
import copy
import json
import random
from pathlib import Path
from pprint import pprint

from tqdm import tqdm
from transformers import AutoTokenizer


def init_random_input(len_range: int = 5, value_gen=5) -> list:
    """
    This function generates a list of random integers.
    Args:
        len_range: The length of the list.
        value_gen: The range of the values in the list.
    Returns:
        A list of random integers.
    """
    len_gen = random.randint(2, len_range + 1)
    value_range = list(range(-value_gen, value_gen + 1))
    output = []
    for index in range(len_gen):
        value_gen = random.choice(value_range)
        output.append(value_gen)
    return output


const_integer = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]


def take(input_list: list, n: int) -> list:
    """
    Take the first n elements of a list.
    Args:
        input_list: A list of elements.
        n: The number of elements to take.
    Returns:
        A list of the first n elements of input_list.
    Raises:
        ValueError: If n is negative.
    """
    return input_list[:n]


def drop(input_list: list, n: int) -> list:
    """
    Drop the first n elements from the input list.
    Args:
        input_list: A list of elements.
        n: The number of elements to drop.
    Returns:
        A list of elements.
    Raises:
        ValueError: If n is greater than the length of input_list.
    """
    return input_list[n:]


def minimum(input_list: list) -> int:
    """
    Returns the minimum value in a list.
    Args:
        input_list: A list of integers.
    Returns:
        The minimum value in the list.
    Raises:
        ValueError: If the list is empty.
    """
    return min(input_list)


def maximum(input_list: list) -> int:
    """
    Finds the maximum value in a list.
    Args:
        input_list: A list of integers.
    Returns:
        The maximum value in the list.
    Raises:
        ValueError: If the list is empty.
    """
    return max(input_list)


def reverse(input_list: list) -> list:
    """
    Reverses the order of the elements in the list.
    Args:
        input_list: The list to be reversed.
    Returns:
        The reversed list.
    """
    return input_list[::-1]


def sort_asc(input_list: list) -> list:
    """
    Sort a list in ascending order.
    Args:
        input_list: A list of numbers.
    Returns:
        A list of numbers sorted in ascending order.
    Raises:
        TypeError: If input_list is not a list.
    """
    return sorted(input_list)


def sort_des(input_list: list) -> list:
    """
    Sort a list in descending order.
    Args:
        input_list: A list of numbers.
    Returns:
        A list of numbers sorted in descending order.
    Raises:
        TypeError: If input_list is not a list.
    """
    return sorted(input_list, reverse=True)


def add_n(input_list: list, n: int) -> list:
    """
    Add n to each element of the input list.
    Args:
        input_list: a list of numbers.
        n: a number.
    Returns:
        A list of each number in input_list, plus n.
    Raises:
        TypeError: if input_list is not a list.
        TypeError: if n is not an int.
    """
    return [x + n for x in input_list]


def sub_n(input_list: list, n: int) -> list:
    """
    Subtract n from each element of the input list.
    Args:
        input_list: The list to be modified.
        n: The number to subtract from each element of the list.
    Returns:
        A new list with the elements of the input list minus n.
    """
    return [x - n for x in input_list]


def mul_n(input_list: list, n: int) -> list:
    """
    Multiply each element in the input list by n.
    Args:
        input_list: A list of integers.
        n: An integer.
    Returns:
        A list of integers.
    """
    return [x * n for x in input_list]


def div_n(input_list: list, n: int) -> list:
    """
    Divides each element in the input list by n.
    Args:
        input_list: A list of numbers.
        n: A number.
    Returns:
        A list of numbers.
    Raises:
        ZeroDivisionError: If n is zero.
    """
    return [x / n for x in input_list]


def expand_copy(input_list: list) -> list:
    """
    Args:
        input_list: a list of integers.
    Returns:
        a list of integers.
    """
    return input_list + input_list


# Main Production Rules for the Toy DSL.
list_manip_dsl = {
    "take": take,
    "drop": drop,
    "reverse": reverse,
    "sort_asc": sort_asc,
    "sort_des": sort_des,
    "add_n": add_n,
    "sub_n": sub_n,
    "mul_n": mul_n,
    "expand_copy": expand_copy,
}


# Use this class to execute programs written in the DSL.
class Interpreter:
    def __init__(self) -> None:
        """
        Initialize the parser.
        """
        self.parser = list_manip_dsl

    def __call__(self, statement_string: str):
        """
        Evaluation Function for the interpreter.
        args:
            statement_string (str) : Statement String
        """
        try:
            return eval(statement_string)  # Adding an exception to unparsable strings
        except:
            return "ERROR"


interpreter = Interpreter()

# TEMPLATE
# This is used to store the input, output and the function template.
# Input : List given as an input to the function.
# function_template : The atomic function in a given DSL Grammar
# Output : Transformed output by applying function on the input.
generation_template = {"function_template": "NONE", "output": "NONE", "input": []}


def gen_take(expr1=None, expr2=None):
    """
    Args:
        expr1: A list of numbers.
        expr2: A number.
    Returns:
        A list of numbers.
    """
    if expr1 == None:
        expr1 = init_random_input()
    if expr2 == None:
        expr2 = random.choice(range(1, len(expr1) - 1))

    formatted_fn = f"take({expr1},{expr2})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1, expr2]
    return template


def gen_drop(expr1=None, expr2=None):
    """
    Drop the first n elements of a list.
    Args:
        expr1: A list of elements.
        expr2: The number of elements to drop.
    Returns:
        A list of elements.
    """
    if expr1 == None:
        expr1 = init_random_input()
    if expr2 == None:
        expr2 = random.choice(range(1, len(expr1) - 1))

    formatted_fn = f"drop({expr1},{expr2})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1, expr2]
    return template


def gen_minimum(expr1=None):
    """
    Args:
        expr1: A string representing a mathematical expression.
    Returns:
        A dictionary containing the following keys:
            "function_template": A string representing a mathematical expression.
            "output": The result of evaluating the "function_template".
            "input": A list of strings representing mathematical expressions.
    """
    if expr1 == None:
        expr1 = init_random_input()

    formatted_fn = f"minimum({expr1})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1]
    return template


def gen_maximum(expr1=None):
    """
    Args:
        expr1: A valid python expression.
    Returns:
        A dictionary containing the following keys:
            "function_template": A string representing the function to be generated.
            "output": The output of the function.
            "input": The input of the function.
    """
    if expr1 == None:
        expr1 = init_random_input()

    formatted_fn = f"maximum({expr1})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1]
    return template


def gen_reverse(expr1=None):
    """
    Args:
        expr1: A string representing a valid python expression.
    Returns:
        A dictionary containing the following keys:
            "function_template": A string representing a valid python function.
            "output": The output of the function.
            "input": The input of the function.
    """
    if expr1 == None:
        expr1 = init_random_input()

    formatted_fn = f"reverse({expr1})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1]
    return template


def gen_sort_asc(expr1=None):
    """
    Args:
        expr1: A list of numbers.
    Returns:
        A list of numbers sorted in ascending order.
    """
    if expr1 == None:
        expr1 = init_random_input()

    formatted_fn = f"sort_asc({expr1})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1]
    return template


def gen_sort_des(expr1=None):
    """
    Args:
        expr1: A list of numbers.
    Returns:
        A list of numbers sorted in descending order.
    """
    if expr1 == None:
        expr1 = init_random_input()

    formatted_fn = f"sort_des({expr1})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1]
    return template


def gen_add_n(expr1=None, expr2=None):
    """
    Args:
        expr1: The first expression.
        expr2: The second expression.
    Returns:
        The result of the addition.
    """
    if expr1 == None:
        expr1 = init_random_input()
    if expr2 == None:
        expr2 = random.choice(const_integer)

    formatted_fn = f"add_n({expr1},{expr2})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1, expr2]
    return template


def gen_sub_n(expr1=None, expr2=None):
    """
    Args:
        expr1: The first expression.
        expr2: The second expression.
    Returns:
        The result of the subtraction.
    """
    if expr1 == None:
        expr1 = init_random_input()
    if expr2 == None:
        expr2 = random.choice(const_integer)

    formatted_fn = f"sub_n({expr1},{expr2})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1, expr2]
    return template


def gen_mul_n(expr1=None, expr2=None):
    """
    Args:
        expr1: A valid python expression.
        expr2: A valid python expression.
    Returns:
        A dictionary containing the following keys:
            "function_template": A string representing the function template.
            "output": The output of the function.
            "input": A list of the input values.
    """
    if expr1 == None:
        expr1 = init_random_input()
    if expr2 == None:
        expr2 = random.choice(const_integer)

    formatted_fn = f"mul_n({expr1},{expr2})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1, expr2]
    return template


def gen_div_n(expr1=None, expr2=None):
    """
    Args:
        expr1: a valid python expression.
        expr2: a valid python expression.
    Returns:
        A dictionary containing the following keys:
            "function_template": a string representing the function template.
            "output": the output of the function.
            "input": a list of the function's input.
    """
    if expr1 == None:
        expr1 = init_random_input()
    if expr2 == None:
        expr2 = random.choice(const_integer)

    formatted_fn = f"div_n({expr1},{expr2})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1, expr2]
    return template


def gen_expand_copy(expr1=None, expr2=None):
    """
    Args:
        expr1: A string representing a mathematical expression.
        expr2: An integer representing the number of times to copy the expression.
    Returns:
        A string representing the expanded expression.
    """
    if expr1 == None:
        expr1 = init_random_input()
    if expr2 == None:
        expr2 = random.choice(range(1, 3))

    formatted_fn = f"expand_copy({expr1},{expr2})"
    template = copy.copy(generation_template)
    template["function_template"] = formatted_fn
    template["output"] = interpreter(formatted_fn)
    template["input"] = [expr1, expr2]
    return template


list_manip_dsl_gen = {
    "take": gen_take,
    "drop": gen_drop,
    "minimum": gen_minimum,
    "maximum": gen_maximum,
    "reverse": gen_reverse,
    "sort_asc": gen_sort_asc,
    "sort_des": gen_sort_des,
    "add_n": gen_add_n,
    "sub_n": gen_sub_n,
    "mul_n": gen_mul_n,
    "div_n": gen_div_n,
    "expand_copy": gen_expand_copy,
}


class Sampler:
    def __init__(
        self,
        max_sample_length: int = 5,
        code_sep: str = ";",
        interpreter_sep: str = "->",
    ):
        """
        Initialize the class.
        Args:
            max_sample_length: The maximum length of the sample.
            code_sep: The separator between the code.
            interpreter_sep: The separator between the interpreter.
        """
        self.max_sample_length = max_sample_length
        self.parser = Interpreter()
        self.production_list = list_manip_dsl
        self.production_idt = [i for i in self.production_list.keys()]
        self.production_gen_list = list_manip_dsl_gen
        self.code_sep = code_sep
        self.interpreter_sep = interpreter_sep

    def sample_production(self, gen_length: int = 5):
        """
        Generate a list of hash functions.
        Args:
            gen_length: The length of the list of hash functions.
        Returns:
            A list of hash functions.
        """
        init_flag = True
        hash_functions = []
        if gen_length == None:
            gen_length = self.max_sample_length

        for ind in range(gen_length):
            if init_flag:
                random_chosen_function = random.choice(self.production_idt)
                generated_function = self.production_gen_list[random_chosen_function]()
                hash_functions.append(generated_function)
                init_flag = False
            else:
                random_chosen_function = random.choice(self.production_idt)
                generated_function = self.production_gen_list[random_chosen_function](
                    hash_functions[-1]["function_template"]
                )
                if generated_function["output"] == "ERROR":
                    break
                hash_functions.append(generated_function)

        return hash_functions


def create_synthetic_dataset(size: int, io_size=3) -> dict:
    """
    Args:
        size: The number of samples to generate.
        io_size: The number of inputs and outputs to generate.
    Returns:
        A list of dictionaries, each dictionary contains the following keys:
            "input": The input string.
            "output": The output string.
            "io_inp": The input list.
            "io_out": The output list.
    """
    output_list = []
    sampler = Sampler()
    for i in tqdm(range(size)):
        try:
            sampled = sampler.sample_production()
            inp = sampled[0]["input"][0]
            out = sampled[-1]["output"]
            function = sampled[-1]["function_template"]
            prompt_inp = f"Input: {inp} Output: {out} Function:"
            prompt_out = function
            if out != [] and out != "ERROR":
                output_list.append(
                    {
                        "input": prompt_inp,
                        "output": prompt_out,
                        "io_inp": inp,
                        "io_out": out,
                    }
                )
        except:
            pass

    return output_list


def write_to_json(data: dict, file_name: str):
    """
    Write data to a json file.
    Args:
        data: The data to write.
        file_name: The file name.
    """
    with open(file_name, "w") as f:
        json.dump(data, f, indent=2)


def basic_stats(dataset, tokenizer):
    """
    Basic stats to calculate the token length of the dataset.
    """
    length_list = []
    for examples in tqdm(dataset):
        datapoint = tokenizer(examples["input"] + " " + examples["output"] + "<|endoftext|>")
        length_list.append(len(datapoint["input_ids"]))
    return {
        "max": max(length_list),
        "min": min(length_list),
        "mean": sum(length_list) / len(length_list),
    }


if __name__ == "__main__":
    # sampler = Sampler()
    # pprint(sampler.sample_production())
    # pprint(interpreter("div_n(reverse([-2, -5, -4]),1)"))
    train_data = create_synthetic_dataset(2000000)
    test_data = create_synthetic_dataset(2_000)
    print(f"Train data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")
    Path("dataset").mkdir(parents=True, exist_ok=True)
    write_to_json(train_data, "dataset/train.json")
    write_to_json(test_data, "dataset/test.json")
