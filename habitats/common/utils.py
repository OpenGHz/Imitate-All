def get_values_by_names(sub_names: tuple, all_names: tuple, all_values: tuple) -> tuple:
    """根据子名称列表获取所有值列表中对应的值列表，返回子值列表"""
    sub_values = [0.0 for _ in range(len(sub_names))]
    for i, name in enumerate(sub_names):
        sub_values[i] = all_values[all_names.index(name)]
    return tuple(sub_values)
