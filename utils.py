def extract_side(value):
    if 'OS' in value:
        return 'LEFT'
    elif 'OD' in value:
        return 'RIGHT'
    else:
        return None
    
def gen_next_date(value):
    if value == 'P20':
        return 'P40'
    elif value == 'P40':
        return 'P60'
    elif value == 'P60':
        return 'P90'
    elif value == 'P90':
        return 'P120'
    elif value == 'P120':
        return 'P150'
    elif value == 'P150':
        return 'P158'
    else:
        return None
    
def remove_tail(df, count=2):
    df = df.sort_values(by='pos')
    return df.iloc[:-count]

def gen_position(path):
    return int(path.split('.')[-2].split('_')[-1])