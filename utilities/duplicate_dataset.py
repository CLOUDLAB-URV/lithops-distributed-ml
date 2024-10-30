import sys

input_file = sys.argv[1]
output_file = sys.argv[2]
repeat_times = int(sys.argv[3])
input = open(input_file)
output = open(output_file, "w")
content = input.read()

for i in range(repeat_times):
    output.write(content)
    if i < (repeat_times-1):
        output.write('\n')