from django.urls import path

from . import views


urlpatterns = {
    path('treinar-modelo', views.ProcessamentoModeloMachineLearningView.as_view(), name='treinar_modelo'),
    path('prever', views.PrevisaoView.as_view(), name='prever'),
}
