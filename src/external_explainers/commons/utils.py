def to_valid_latex(string, is_bold: bool = False) -> str:
    """
    Convert a string to a valid latex string.
    :param string: The input string.
    :param is_bold: Whether the string should be bold.
    :return: The latex string.
    """
    latex_string = str(string)  # unicode_to_latex(string)
    # Choose the space character based on whether the string should be bold.
    space = r'\ ' if is_bold else ' '
    # Escape special characters and replace spaces with the chosen space character.
    final_str = latex_string.replace("&", "\\&").replace("#", "\\#").replace(' ', space).replace("_", space)
    return final_str


def to_valid_latex_with_escaped_dollar_char(string, is_bold: bool = False) -> str:
    """
    Convert a string to a valid latex string.
    This function adds the $ character to the list of characters that are escaped, while the function to_valid_latex
    does not escape the $ character. Other than that, the two functions are identical.
    :param string: The input string.
    :param is_bold: Whether the string should be bold.
    :return: The latex string.
    """
    return to_valid_latex(string, is_bold).replace("$", "\\$")