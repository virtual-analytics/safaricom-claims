"""
WSGI config for myproject project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os
import sys
import traceback

try:
    from django.core.wsgi import get_wsgi_application

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')

    application = get_wsgi_application()
except Exception as e:
    print("WSGI startup error:", e, file=sys.stderr)
    traceback.print_exc()
    raise
