from django import template
register = template.Library()

@register.filter
def equals(val1, val2):
    return str(val1) == str(val2)

@register.filter
def get_item(dictionary, key):
    return dictionary.get(key) 