def normalize(column, col_min, col_max):
    return (column - col_min) / (col_max - col_min)
