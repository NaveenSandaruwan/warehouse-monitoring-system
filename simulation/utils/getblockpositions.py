def getblockposi():
    coordinates_to_block = [(12, 3), (11, 1),(12, 5)]

    # Read the file and get coordinates
    with open('camprocess/real_time_updates.txt', 'r') as file:
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