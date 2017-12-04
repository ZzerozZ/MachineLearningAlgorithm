# Doc data tu file:
data = open("ex1data2.txt", "r").readlines()
for i in range(len(data)):
    data[i] = data[i].replace('\t', ' ')
    data[i] = data[i].replace('  ', ',')
    data[i] = data[i].replace(' ', ',')
    data[i] = data[i].replace('No', '0')
    data[i] = data[i].replace('Yes', '1')

write_down = open("ex1data2.txt", "w")
for i in range(len(data)):
    write_down.write(data[i])

