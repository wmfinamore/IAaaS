# Native Python Library
from decimal import Decimal

# Django libraries
from django.db import models
from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator

# h2o library
# pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o
import h2o
from h2o.automl import H2OAutoML

# Pandas Library
import pandas as pd


# Model that represent the database table for that the register
# processing data become persisted
class ModeloMachineLearningProcessado(models.Model):
    model_id = models.TextField('Identificador do modelo',
                                null=True, blank=True)
    auc = models.DecimalField('aux', max_digits=10, decimal_places=6,
                              null=True, blank=True)
    logloss = models.DecimalField('logloss', max_digits=10, decimal_places=6,
                                  null=True, blank=True)
    aucpr = models.DecimalField('aucpr', max_digits=10, decimal_places=6,
                                null=True, blank=True)
    mean_per_class_error = models.DecimalField('mean_per_class_error',
                                               max_digits=10, decimal_places=6,
                                               null=True, blank=True)
    rmse = models.DecimalField('rmse', max_digits=10, decimal_places=6,
                               null=True, blank=True)
    mse = models.DecimalField('mse', max_digits=10, decimal_places=6,
                              null=True, blank=True)
    binario_modelo = models.FileField('Binário do modelo ML',
                                      upload_to='binario_modelo_ml',
                                      null=True, blank=True)

# Model that represent the database table for the process
class ProcessamentoModeloMachineLearning(models.Model):
    data = models.DateTimeField('Data e hora do processamento', auto_now_add=True)
    dados_csv = models.FileField('Arquivo CSV', upload_to='arquivos_csv')
    classe = models.CharField('Classe', max_length=30, help_text='Variável dependente')
    variaveis_independentes = models.TextField('Variáveis independentes', null=True)
    tempo_maximo = models.PositiveIntegerField('Tempo máximo em segundos',
                                               validators=[MinValueValidator(settings.TEMPO_MINIMO_PROCESSAMENTO),
                                                           MaxValueValidator(settings.TEMPO_MAXIMO_PROCESSAMENTO)],)
    modelos_processados = models.ManyToManyField('rest.ModeloMachineLearningProcessado', related_name='processamentos')

    class Meta:
        ordering = ['-data']

    def processar(self):
        """
            Método que processa o Auto Machine Learning e guarda os resultados do melhor
            modelo raqueado nos atributos da ORM Django ProcessamentoModeloMachineLearning
        """

        h2o.init()
        # Importa dados do CSV que foi gravado no atributo dados_csv do modelo do ORM
        imp = pd.read_csv(self.dados_csv, sep=";")

        # Identifica dinamicamente as colunas do arquivo CSV
        colunas = imp.columns.tolist()

        # Seleciona as variáveis independentes de forma excludentes, considerando a classe
        variaveis_independentes = [coluna for coluna in colunas if coluna != self.classe]
        self.variaveis_independentes = ','.join(variaveis_independentes)
        self.save()

        # Divide em treino e teste
        imp = h2o.H2OFrame(imp)
        treino, teste = imp.split_frame(ratios=[.7])

        # Transforma a variável dependente em fator
        treino[self.classe] = treino[self.classe].asfactor()
        teste[self.classe] = teste[self.classe].asfactor()

        # Auto ML
        # Busca o modelo valor gravado no atributo tempo_maximo em segundos,
        # podemos em vez disso definir max_models
        modelo_automl = H2OAutoML(max_runtime_secs=self.tempo_maximo, sort_metric='AUC')
        modelo_automl.train(y=self.classe, training_frame=treino)

        # Ranking dos melhores AutoML
        ranking = modelo_automl.leaderboard
        ranking = ranking.as_data_frame()

        # Salva os resultados dos modelos rankeados associados ao processamento
        for i in range(0, len(ranking)-1):
            modelo_processado = ModeloMachineLearningProcessado()
            modelo_processado.model_id = ranking['model_id'].iloc[i]
            modelo_processado.auc = ranking['auc'].iloc[i].astype(Decimal)
            modelo_processado.logloss = ranking['logloss'].iloc[i].astype(Decimal)
            modelo_processado.aucpr = ranking['aucpr'].iloc[i].astype(Decimal)
            modelo_processado.mean_per_class_error = ranking['mean_per_class_error'].iloc[i].astype(Decimal)
            modelo_processado.rmse = ranking['rmse'].iloc[i].astype(Decimal)
            modelo_processado.mse = ranking['mse'].iloc[i].astype(Decimal)
            modelo = h2o.get_model(modelo_processado.model_id)
            modelo_processado.binario_modelo.name = h2o.save_model(modelo,
                                                                   path="%s/modelo" % settings.MEDIA_ROOT,
                                                                   force=True)
            modelo_processado.save()
            self.modelos_processados.add(modelo_processado)
