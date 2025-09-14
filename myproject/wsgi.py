"""
WSGI config for myproject project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os
import sys
import traceback
import time

try:
    from django.core.wsgi import get_wsgi_application

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')

    def debug_application(environ, start_response):
        print(f"[WSGI DEBUG] Request: {environ.get('REQUEST_METHOD')} {environ.get('PATH_INFO')} at {time.strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
        return application(environ, start_response)

    application = get_wsgi_application()
    print("[WSGI DEBUG] WSGI application started successfully", file=sys.stderr)
    application = debug_application
except Exception as e:
    print("WSGI startup error:", e, file=sys.stderr)
    traceback.print_exc()
    raise
