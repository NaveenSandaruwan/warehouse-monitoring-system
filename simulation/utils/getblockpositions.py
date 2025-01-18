def getblockposi():
    coordinates_to_block = [(0, 0), (11, 1)]

    # Read the file and get coordinates
    with open('simulation/camprocess/real_time_updates.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line:
                coordinates = eval(line)
                for coord in coordinates:
                    if coord not in coordinates_to_block:
                        coordinates_to_block.append(coord)

    print(coordinates_to_block)
    return coordinates_to_block