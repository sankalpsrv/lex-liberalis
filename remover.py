import re

def remove_text(input_string, start_constant, end_constant):
    pattern = re.escape(start_constant) + r".*?" + re.escape(end_constant)
    output_string = re.sub(pattern, '', input_string)
    return output_string
