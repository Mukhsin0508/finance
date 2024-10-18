from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.urls import path, include
from rest_framework.documentation import include_docs_urls
from rest_framework.schemas import get_schema_view
from rest_framework.renderers import CoreJSONRenderer


schema_view = get_schema_view(
    title="Your API Title",
    renderer_classes=[CoreJSONRenderer]
)

admin.site.site_header = "nohunt.uz Administration"
admin.site.index_title = "Site Administration"
admin.site.site_title = "nohunt.uz Administration"

prefix = settings.BACKEND_PROXY_URL

urlpatterns = [
    path(prefix + "", include("finance.urls")),

    path(prefix + "admin/", admin.site.urls),
    path(prefix + "docs/", include_docs_urls(title="Fn")),
    path('schema/', schema_view, name='api-schema'),
]

urlpatterns += staticfiles_urlpatterns()

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
