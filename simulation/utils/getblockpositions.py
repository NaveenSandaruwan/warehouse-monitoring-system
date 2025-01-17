def getblockposi():
    coordinates_to_block = [(0, 0), (11, 1)]
    # Coordinates to block (set to 1)
    row_1 = (13, 6)
    row_2 = (13, 4)
    row_3 = (13, 2)

    # Read the file and get row person count
    with open('simulation/camprocess/counts.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Row 1" in line:
                count = int(line.split(":")[1].strip().split()[0])
                if count >= 3:
                    coordinates_to_block.append(row_1)
            if "Row 2" in line:
                count = int(line.split(":")[1].strip().split()[0])
                if count >= 3:
                    coordinates_to_block.append(row_2)
            if "Row 3" in line:
                count = int(line.split(":")[1].strip().split()[0])
                if count >= 3:
                    coordinates_to_block.append(row_3)

    print(coordinates_to_block)
    return coordinates_to_block