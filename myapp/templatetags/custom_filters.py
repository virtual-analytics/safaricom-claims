# myapp/templatetags/custom_filters.py
from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Template filter to access dictionary items by key"""
    return dictionary.get(key)