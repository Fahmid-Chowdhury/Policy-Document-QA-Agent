from django.urls import path
from .views import health, index_endpoint, ask_endpoint, ask_json_endpoint, warmup_endpoint

urlpatterns = [
    path("health/", health),

    # versioned API
    path("v1/index", index_endpoint),
    path("v1/ask", ask_endpoint),
    path("v1/ask_json", ask_json_endpoint),
    path("v1/warmup", warmup_endpoint),
]
