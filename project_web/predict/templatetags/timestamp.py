from django import template
register = template.Library()

import time

@register.simple_tag
def timestamp():
    now = time.time()
    return now