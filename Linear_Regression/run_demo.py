data = open("result.txt", "r").readlines()[0].replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').split(' ')
test_case = [82.50, 72.25, 20.37, 0.05]
print sum([test_case[i] * float(data[i]) for i in range(len(data))])
