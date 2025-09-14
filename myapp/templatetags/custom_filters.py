from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Template filter to access dictionary items by key"""
    return dictionary.get(key)

@register.filter
def multiply(value, arg):
    """Multiply the value by the arg"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def abs_val(value):
    """Return the absolute value"""
    try:
        return abs(float(value))
    except (ValueError, TypeError):
        return 0
    
@register.filter(name="abs")
def abs_val(value):
    try:
        return abs(float(value))
    except (ValueError, TypeError):
        return 0
@register.filter
def isin(value, arg):
    """Check if value (uppercased) is in a comma-separated list of options"""
    if not value:
        return False
    options = [a.strip().upper() for a in arg.split(',')]
    return value.upper() in options


@register.filter
def in_list(value, arg):
    """
    Check if a value is in a comma-separated list.
    Usage: {{ value|in_list:"A,B,C" }}
    """
    if value is None:
        return False
    return str(value).upper() in [x.strip().upper() for x in arg.split(',')]


@register.filter
def dictsum(dict_values, key):
    """
    Sum up a specific key inside a dict of dicts.
    Example: columns.values|dictsum:"null_count"
    """
    return sum(d.get(key, 0) for d in dict_values if isinstance(d, dict))

@register.filter
def absval(value):
    try:
        return abs(float(value))
    except Exception:
        return value
    
@register.filter
def to_matrix(value):
    """
    Convert a dict-of-dicts (correlation matrix) into a list of lists
    for Chart.js.
    """
    if not isinstance(value, dict):
        return []
    keys = list(value.keys())
    matrix = []
    for k in keys:
        row = [value[k].get(inner, 0) for inner in keys]
        matrix.append(row)
    return matrix

@register.filter
def get_keys(value):
    """Return the keys of a dictionary as a list"""
    if isinstance(value, dict):
        return list(value.keys())
    return []



@register.filter
def multiply(value, arg):
    """Multiply the value by the argument"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def div(value, arg):
    try:
        return float(value) / float(arg) if arg else 0
    except (ValueError, ZeroDivisionError):
        return 0

@register.filter
def mul(value, arg):
    try:
        return float(value) * float(arg)
    except ValueError:
        return 0
    
@register.filter
def get_item(dictionary, key):
    """Get item from dict safely in templates."""
    if dictionary and key in dictionary:
        return dictionary.get(key)
    return None

@register.filter
def get_item(dictionary, key):
    return dictionary.get(key, [])



@register.filter
def get_item(dictionary, key):
    """Get a value from a dictionary using a key"""
    return dictionary.get(key) if dictionary else None

@register.filter
def zip(lists, other_list):
    """Zip two lists together"""
    return zip(lists, other_list)