import random
from collections import defaultdict
from shapely.geometry import LineString


def group_touching_lines(lines):
    endpoint_to_lines = defaultdict(list)

    # Populate the dictionary with lines grouped by endpoints
    for line in lines:
        endpoints = [line.coords[0], line.coords[-1]]
        for endpoint in endpoints:
            endpoint_to_lines[endpoint].append(line)

    # Keep track of lines that have been assigned to a group
    assigned_lines = set()

    grouped_lines = []

    for line in lines:
        if line in assigned_lines:
            continue

        group = []
        stack = [line]
        assigned_lines.add(line)

        while stack:
            current_line = stack.pop()
            group.append(current_line)

            endpoints = [current_line.coords[0], current_line.coords[-1]]

            # Look for lines sharing the same endpoints and add them to the stack
            for endpoint in endpoints:
                for connected_line in endpoint_to_lines[endpoint]:
                    if connected_line not in assigned_lines:
                        stack.append(connected_line)
                        assigned_lines.add(connected_line)

        grouped_lines.append(group)

    return grouped_lines


def generate_random_line(length=10):
    x1 = random.uniform(0, length)
    y1 = random.uniform(0, length)
    x2 = random.uniform(0, length)
    y2 = random.uniform(0, length)
    return LineString([(x1, y1), (x2, y2)])

# Generate a larger testing dataset
num_lines = 50
test_lines = [generate_random_line() for _ in range(num_lines)]

grouped_touching_lines = group_touching_lines(test_lines)

for group_index, group in enumerate(grouped_touching_lines):
    print(f"Group {group_index + 1}: {group}")
