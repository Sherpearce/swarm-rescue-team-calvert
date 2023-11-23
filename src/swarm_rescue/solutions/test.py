def find_opposing_indices(index, num_vertices):
    if index < 0 or index >= num_vertices:
        raise ValueError("Index out of range")

    # Calculate the two opposing indices
    opposing_index1 = (index + num_vertices // 2) % num_vertices
    return (opposing_index1%num_vertices, (opposing_index1 + 1)%num_vertices)

# Example usage:
num_vertices = 5
given_index = 1  # Replace this with the desired index
opposing_indices = find_opposing_indices(given_index, num_vertices)
print(find_opposing_indices(0, num_vertices))
print(find_opposing_indices(1, num_vertices))
print(find_opposing_indices(2, num_vertices))
print(find_opposing_indices(3, num_vertices))
print(find_opposing_indices(4, num_vertices))
