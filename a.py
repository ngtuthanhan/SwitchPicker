# Input string of floats separated by spaces
float_string = "3.14 2.718 1.618 0.577 1.414"

# Split the string into individual substrings using 'split()'
float_list = float_string.split()

# Convert each substring to a float using list comprehension
float_list = [float(num) for num in float_list]

# Output the list of floats
print(float_list)
